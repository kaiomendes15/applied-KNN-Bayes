"""Leitor de arquivos ARFF (formato do OpenML/Weka) sem pandas nem bibliotecas de ML.

Duas etapas:
1. `load_arff` lê o arquivo e devolve o cabeçalho (atributos) e as linhas como valores Python.
2. `to_numpy` converte para matrizes numpy: atributos nominais viram one-hot,
   o alvo nominal vira índice inteiro de classe e o alvo numérico vira float.

Suporta o formato ARFF denso: comentários (%), valores entre aspas simples ou
duplas, atributos numeric/real/integer, nominais ({...}), string e valores ausentes (?).
"""

from dataclasses import dataclass, field

import numpy as np

NUMERIC_TYPES = {"numeric", "real", "integer"}


@dataclass
class Attribute:
    name: str
    type: str  # "numeric", "nominal", "string" ou "date"
    values: list = field(default_factory=list)  # categorias, apenas para nominais

    @property
    def is_numeric(self):
        return self.type == "numeric"

    @property
    def is_nominal(self):
        return self.type == "nominal"


@dataclass
class ArffDataset:
    relation: str
    attributes: list
    rows: list  # lista de linhas; cada valor é float, str ou None (ausente)

    def attribute_index(self, name):
        for i, attr in enumerate(self.attributes):
            if attr.name == name:
                return i
        raise KeyError(f"Atributo '{name}' não existe em '{self.relation}'")


def _split_values(text):
    """Separa uma linha por vírgulas, respeitando aspas simples/duplas e escapes (\\)."""
    tokens, current, quote, was_quoted = [], [], None, False
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\" and i + 1 < len(text):
                current.append(text[i + 1])
                i += 1
            elif ch == quote:
                quote = None
            else:
                current.append(ch)
        elif ch in "'\"":
            # Espaços antes da aspa de abertura não fazem parte do valor.
            current = [c for c in current if not c.isspace()]
            quote, was_quoted = ch, True
        elif ch == ",":
            tokens.append(_finish_token(current, was_quoted))
            current, was_quoted = [], False
        elif not (was_quoted and ch.isspace()):  # ignora espaços após a aspa de fechamento
            current.append(ch)
        i += 1
    if quote:
        raise ValueError(f"Aspas não fechadas em: {text!r}")
    tokens.append(_finish_token(current, was_quoted))
    return tokens


def _finish_token(chars, was_quoted):
    token = "".join(chars)
    # Espaços fora das aspas não fazem parte do valor; dentro das aspas, sim.
    # Valores entre aspas são marcados para que "'?'" não seja confundido com ausente.
    return (token if was_quoted else token.strip(), was_quoted)


def _parse_attribute(line, line_number):
    rest = line[len("@attribute"):].strip()

    # O nome pode vir entre aspas e conter espaços.
    if rest[0] in "'\"":
        end = rest.index(rest[0], 1)
        name, type_spec = rest[1:end], rest[end + 1:].strip()
    else:
        name, _, type_spec = rest.partition(" ")
        if not type_spec:
            name, _, type_spec = rest.partition("\t")
        type_spec = type_spec.strip()

    if type_spec.startswith("{"):
        if not type_spec.endswith("}"):
            raise ValueError(f"Linha {line_number}: lista nominal sem '}}'")
        values = [value for value, _ in _split_values(type_spec[1:-1])]
        return Attribute(name, "nominal", values)

    kind = type_spec.split()[0].lower()
    if kind in NUMERIC_TYPES:
        return Attribute(name, "numeric")
    if kind in ("string", "date"):
        return Attribute(name, kind)
    raise ValueError(f"Linha {line_number}: tipo de atributo não suportado: {type_spec!r}")


def _parse_value(token, attr, line_number):
    text, was_quoted = token
    if text == "?" and not was_quoted:
        return None
    if attr.is_numeric:
        try:
            return float(text)
        except ValueError:
            raise ValueError(
                f"Linha {line_number}: valor {text!r} não é numérico (atributo '{attr.name}')"
            ) from None
    if attr.is_nominal and text not in attr.values:
        raise ValueError(
            f"Linha {line_number}: categoria {text!r} não declarada no atributo '{attr.name}'"
        )
    return text


def load_arff(path):
    """Lê um arquivo ARFF denso e devolve um `ArffDataset`."""
    relation, attributes, rows = None, [], []
    in_data = False

    with open(path, encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("%"):
                continue

            if not in_data:
                lower = line.lower()
                if lower.startswith("@relation"):
                    relation = line[len("@relation"):].strip().strip("'\"")
                elif lower.startswith("@attribute"):
                    attributes.append(_parse_attribute(line, line_number))
                elif lower.startswith("@data"):
                    in_data = True
                else:
                    raise ValueError(f"Linha {line_number}: cabeçalho inesperado: {line!r}")
                continue

            if line.startswith("{"):
                raise ValueError("Formato ARFF esparso não é suportado")
            tokens = _split_values(line)
            if len(tokens) != len(attributes):
                raise ValueError(
                    f"Linha {line_number}: {len(tokens)} valores, esperados {len(attributes)}"
                )
            rows.append([_parse_value(t, a, line_number) for t, a in zip(tokens, attributes)])

    if not in_data:
        raise ValueError(f"Seção @data não encontrada em {path}")
    return ArffDataset(relation, attributes, rows)


def to_numpy(dataset, target=None, drop_first=False, ignore=()):
    """Converte um `ArffDataset` em (X, y, feature_names, class_names).

    - target: nome do atributo alvo (padrão: o último atributo).
    - ignore: nomes de atributos que ficam fora de X (identificadores, por exemplo).
    - Atributos numéricos viram uma coluna float (ausentes viram np.nan).
    - Atributos nominais viram colunas one-hot "atributo=categoria". As categorias
      vêm do cabeçalho, então a codificação é a mesma em qualquer fold, sem vazamento.
      Valor ausente gera np.nan em todas as colunas daquele atributo.
    - drop_first=True remove a primeira categoria de cada atributo nominal, evitando
      colunas linearmente dependentes (útil na Regressão Linear e no Bayes multivariado).
    - Alvo nominal: y contém índices inteiros e class_names[i] é o nome da classe i.
      Alvo numérico: y é float e class_names é None.
    """
    target_idx = dataset.attribute_index(target) if target else len(dataset.attributes) - 1
    target_attr = dataset.attributes[target_idx]
    ignored = {dataset.attribute_index(name) for name in ignore}

    columns, feature_names = [], []
    for j, attr in enumerate(dataset.attributes):
        if j == target_idx or j in ignored:
            continue
        values = [row[j] for row in dataset.rows]

        if attr.is_numeric:
            columns.append([np.nan if v is None else v for v in values])
            feature_names.append(attr.name)
        elif attr.is_nominal:
            categories = attr.values[1:] if drop_first else attr.values
            for category in categories:
                columns.append([np.nan if v is None else float(v == category) for v in values])
                feature_names.append(f"{attr.name}={category}")
        else:
            raise ValueError(f"Atributo '{attr.name}' do tipo {attr.type} não pode virar número")

    X = np.array(columns, dtype=float).T if columns else np.empty((len(dataset.rows), 0))

    targets = [row[target_idx] for row in dataset.rows]
    if None in targets:
        raise ValueError(f"O alvo '{target_attr.name}' tem valores ausentes")

    if target_attr.is_nominal:
        class_names = list(target_attr.values)
        y = np.array([class_names.index(v) for v in targets], dtype=int)
    elif target_attr.is_numeric:
        class_names = None
        y = np.array(targets, dtype=float)
    else:
        raise ValueError(f"Alvo '{target_attr.name}' do tipo {target_attr.type} não suportado")

    return X, y, feature_names, class_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resumo de arquivos ARFF.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--target", help="atributo alvo (padrão: o último)")
    parser.add_argument("--ignore", default="", help="atributos a ignorar, separados por vírgula")
    args = parser.parse_args()
    ignore = [name for name in args.ignore.split(",") if name]

    for path in args.paths:
        ds = load_arff(path)
        X, y, names, classes = to_numpy(ds, target=args.target, ignore=ignore)
        target_name = args.target or ds.attributes[-1].name
        predictors = [a for a in ds.attributes if a.name != target_name and a.name not in ignore]
        n_nominal = sum(a.is_nominal for a in predictors)
        print(f"== {path} ({ds.relation})")
        print(f"   amostras: {X.shape[0]} | atributos originais: {len(predictors)} "
              f"({n_nominal} nominais) | colunas após one-hot: {X.shape[1]}")
        print(f"   alvo: '{target_name}'", end="")
        if classes:
            counts = np.bincount(y, minlength=len(classes))
            print(" | classes: " + ", ".join(f"{c}={k}" for c, k in zip(classes, counts)))
        else:
            print(f" | min={y.min():.2f} média={y.mean():.2f} max={y.max():.2f}")
        if ignore:
            print(f"   ignorados: {', '.join(ignore)}")
        print(f"   valores ausentes em X: {int(np.isnan(X).sum())}")
