"""Script final do projeto (Membro 4): roda TODOS os modelos nos dois datasets
e gera as duas tabelas comparativas.

Junta as entregas de todos os membros:
- Membro 1: runner (z-score, k-fold, métricas e tempos) e geração das tabelas.
- Membro 2: kNN (classificação e regressão), nas duas distâncias.
- Membro 3: Bayesiano univariado e multivariado.
- Membro 4: Regressão Linear Múltipla (esta entrega) e esta integração.

Tabelas geradas:
- Classificação (credit-g): kNN (Euclidiana), kNN (Manhattan),
  Bayesiano Univariado, Bayesiano Multivariado.
- Regressão (miami_housing): kNN (Euclidiana), kNN (Manhattan),
  Regressão Linear Múltipla.

Detalhe importante do credit-g: o Bayesiano multivariado precisa inverter a
matriz de covariância, então usa `drop_first=True` no leitor ARFF (remove uma
coluna one-hot por atributo, evitando Σ singular). Os demais modelos usam o
one-hot completo. Por isso o credit-g é carregado das duas formas.

Rodar na raiz do projeto: python -m src.run_experiments
"""

from src.arff import load_arff, to_numpy
from src.knn import KNNClassifier, KNNRegressor
from src.bayes import GaussianNaiveBayes, GaussianBayes
from src.linear_regression import LinearRegression
from src.runner import (run_experiment, print_tabela_classificacao,
                        print_tabela_regressao)

K_FOLDS, SEED = 5, 42


def experimentos_classificacao():
    """As 4 linhas da tabela de classificação (credit-g)."""
    # One-hot completo: usado pelo kNN e pelo Bayesiano univariado.
    X, y, _, _ = to_numpy(load_arff("data/classificacao/credit-g.arff"))
    # drop_first: uma coluna one-hot a menos por atributo, para o multivariado
    # conseguir inverter Σ (ver docs/guia_bayes_membro3.md, seção 5.6).
    X_df, y_df, _, _ = to_numpy(load_arff("data/classificacao/credit-g.arff"),
                                drop_first=True)

    resultados = {}
    for metrica, nome in (("euclidean", "kNN (Euclidiana)"),
                          ("manhattan", "kNN (Manhattan)")):
        print(f"credit-g: {nome}...", flush=True)
        resultados[nome] = run_experiment(KNNClassifier(k="auto", metric=metrica),
                                          X, y, K=K_FOLDS, seed=SEED,
                                          task="classification")

    print("credit-g: Bayesiano Univariado...", flush=True)
    resultados["Bayesiano Univariado"] = run_experiment(
        GaussianNaiveBayes(), X, y, K=K_FOLDS, seed=SEED, task="classification")

    print("credit-g: Bayesiano Multivariado...", flush=True)
    resultados["Bayesiano Multivariado"] = run_experiment(
        GaussianBayes(), X_df, y_df, K=K_FOLDS, seed=SEED, task="classification")

    return resultados


def experimentos_regressao():
    """As 3 linhas da tabela de regressão (miami_housing)."""
    # Alvo SALE_PRC; PARCELNO é só identificador, então fica fora de X.
    X, y, _, _ = to_numpy(load_arff("data/regressao/miami_housing.arff"),
                          target="SALE_PRC", ignore=["PARCELNO"])

    resultados = {}
    for metrica, nome in (("euclidean", "kNN (Euclidiana)"),
                          ("manhattan", "kNN (Manhattan)")):
        print(f"miami_housing: {nome}...", flush=True)
        resultados[nome] = run_experiment(KNNRegressor(k="auto", metric=metrica),
                                          X, y, K=K_FOLDS, seed=SEED,
                                          task="regression")

    print("miami_housing: Regressão Linear Múltipla...", flush=True)
    resultados["Regressão Linear Múltipla"] = run_experiment(
        LinearRegression(), X, y, K=K_FOLDS, seed=SEED, task="regression")

    return resultados


if __name__ == "__main__":
    classificacao = experimentos_classificacao()
    regressao = experimentos_regressao()
    print_tabela_classificacao(classificacao)
    print_tabela_regressao(regressao)
