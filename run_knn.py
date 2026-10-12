"""Experimentos do Membro 2: as 4 linhas do kNN (2 distâncias × 2 datasets).

Usa o runner do Membro 1 (z-score, k-fold, métricas e tempos). O k é escolhido por
validação interna dentro do `fit` (k="auto"), então o tempo de treino inclui essa escolha.

Rodar na raiz do projeto: python run_knn.py
"""

from src.arff import load_arff, to_numpy
from src.knn import KNNClassifier, KNNRegressor
from src.runner import (run_experiment, print_tabela_classificacao,
                        print_tabela_regressao)

K_FOLDS, SEED = 5, 42

# --- Classificação: credit-g ---
X, y, _, _ = to_numpy(load_arff("data/classificacao/credit-g.arff"))
classificacao = {}
for metrica, nome in (("euclidean", "kNN (Euclidiana)"), ("manhattan", "kNN (Manhattan)")):
    print(f"credit-g: {nome}...", flush=True)
    classificacao[nome] = run_experiment(KNNClassifier(k="auto", metric=metrica),
                                         X, y, K=K_FOLDS, seed=SEED, task="classification")

# --- Regressão: miami_housing (alvo SALE_PRC; PARCELNO é só identificador) ---
X, y, _, _ = to_numpy(load_arff("data/regressao/miami_housing.arff"),
                      target="SALE_PRC", ignore=["PARCELNO"])
regressao = {}
for metrica, nome in (("euclidean", "kNN (Euclidiana)"), ("manhattan", "kNN (Manhattan)")):
    print(f"miami_housing: {nome}...", flush=True)
    regressao[nome] = run_experiment(KNNRegressor(k="auto", metric=metrica),
                                     X, y, K=K_FOLDS, seed=SEED, task="regression")

print_tabela_classificacao(classificacao)
print_tabela_regressao(regressao)
