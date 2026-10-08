# Leitor ARFF — do arquivo às matrizes numpy

O módulo `src/arff.py` transforma um arquivo `.arff` do OpenML em estruturas que os algoritmos conseguem usar: uma matriz `X` (atributos) e um vetor `y` (alvo).

## Uso

```python
from src.arff import load_arff, to_numpy

dataset = load_arff("data/classificacao/credit-g.arff")
X, y, feature_names, class_names = to_numpy(dataset)
```

No `miami_housing`, o alvo não é o último atributo e a coluna `PARCELNO` é só um identificador, então os dois precisam ser informados:

```python
dataset = load_arff("data/regressao/miami_housing.arff")
X, y, feature_names, _ = to_numpy(dataset, target="SALE_PRC", ignore=["PARCELNO"])
```

| Retorno | credit-g | miami_housing |
|---|---|---|
| `X` | matriz `(1000, 63)` de floats | matriz `(13932, 15)` de floats |
| `y` | vetor de inteiros (`0` = good, `1` = bad) | vetor de floats (preço de venda, `SALE_PRC`) |
| `feature_names` | nome de cada coluna de `X` | nome de cada coluna de `X` |
| `class_names` | `['good', 'bad']` | `None` |

---

## O pipeline em 2 etapas

```
credit-g.arff ──► load_arff ──► ArffDataset ──► to_numpy ──► X, y
  (texto)          (lê)       (listas Python)   (converte)   (numpy)
```

### Etapa 1 — `load_arff`: texto → listas Python

O arquivo tem duas partes, e o leitor trata cada uma de um jeito.

**Cabeçalho:** cada `@attribute` vira um objeto `Attribute`, com nome, tipo e categorias.

```
@attribute 'duration' real                      →  Attribute('duration', 'numeric')
@attribute 'own_telephone' { none, yes}         →  Attribute('own_telephone', 'nominal', ['none', 'yes'])
@attribute 'class' { good, bad}                 →  Attribute('class', 'nominal', ['good', 'bad'])
```

**Dados:** cada linha depois de `@data` vira uma lista de valores Python.

```
'<0',6,'critical/other existing credit',...,yes,good
        ↓
['<0', 6.0, 'critical/other existing credit', ..., 'yes', 'good']
```

As regras de conversão de cada valor são:
- Atributo numérico → `float` (`6` → `6.0`).
- Atributo nominal → `str`. O valor precisa estar entre as categorias do cabeçalho, senão o leitor lança um erro.
- `?` → `None`, que representa valor ausente.

As vírgulas dentro de aspas são respeitadas. `'new car', 'used car'` gera dois valores, não quatro.

### Etapa 2 — `to_numpy`: listas Python → numpy

O último atributo é o **alvo** (`y`). Os demais formam `X`.

**Atributo numérico → 1 coluna, sem mudança**

```
duration: 6.0   →   [6.0]
```

**Atributo nominal → one-hot (1 coluna por categoria)**

Os algoritmos só operam com números, então cada categoria vira uma coluna 0/1:

```
own_telephone ∈ {none, yes}

valor    own_telephone=none   own_telephone=yes
'yes'  →        0                    1
'none' →        1                    0
```

É por isso que o credit-g passa de **20 atributos para 63 colunas**: seus 13 atributos nominais somam 56 categorias, e os 7 numéricos continuam com uma coluna cada.

**Alvo**
- Nominal (classificação): cada classe vira o seu índice → `good` = `0`, `bad` = `1`.
- Numérico (regressão): mantém o valor → `90.0`.

---

## Exemplo completo: 1ª linha do credit-g

```
Linha no arquivo:
'<0', 6, 'critical/other existing credit', radio/tv, 1169, ..., good

load_arff:
['<0', 6.0, 'critical/other existing credit', 'radio/tv', 1169.0, ..., 'good']

to_numpy:
X[0] = [1, 0, 0, 0,     6.0,   0, 0, 0, 0, 1,   ...]
        └─checking_status─┘ duration └─credit_history─┘
         (<0 = 1)                     (critical/... = 1)
y[0] = 0   (good)
```

---

## Opção `drop_first`

```python
X, y, names, classes = to_numpy(dataset, drop_first=True)   # credit-g → (1000, 50)
```

Remove a primeira coluna de cada atributo nominal. A informação não se perde: se todas as colunas restantes forem `0`, a amostra pertence à categoria removida.

```
own_telephone=yes
      1          → yes
      0          → none (categoria removida)
```

**Quando usar:** no **Bayesiano multivariado** e na **Regressão Linear**. Com todas as colunas do one-hot, elas somam sempre 1, o que torna a matriz de covariância (Σ) ou a matriz XᵀX **não inversível**. No kNN e no Bayesiano univariado, não é necessário.

---

## Inspeção rápida pelo terminal

```bash
source .venv/bin/activate
python -m src.arff data/classificacao/credit-g.arff
python -m src.arff data/regressao/miami_housing.arff --target SALE_PRC --ignore PARCELNO
```

```
== data/classificacao/credit-g.arff (german_credit)
   amostras: 1000 | atributos originais: 20 (13 nominais) | colunas após one-hot: 63
   alvo: 'class' | classes: good=700, bad=300
   valores ausentes em X: 0
== data/regressao/miami_housing.arff (x)
   amostras: 13932 | atributos originais: 15 (0 nominais) | colunas após one-hot: 15
   alvo: 'SALE_PRC' | min=72000.00 média=399941.93 max=2650000.00
   ignorados: PARCELNO
   valores ausentes em X: 0
```
