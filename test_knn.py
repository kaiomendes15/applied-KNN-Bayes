"""Testes do kNN (Membro 2). Rodar na raiz do projeto: python test_knn.py"""

import numpy as np

from src.arff import load_arff, to_numpy
from src.knn import KNNClassifier, KNNRegressor, select_k, _pairwise
from src.runner import run_experiment, print_tabela_classificacao, print_tabela_regressao


# =============================================================================
# Teste 1: distâncias (comparadas com o cálculo direto, ponto a ponto)
# =============================================================================

print("=" * 60)
print("TESTE 1: Distâncias Euclidiana e Manhattan")
print("=" * 60)

rng = np.random.RandomState(0)
A, B = rng.randn(7, 5), rng.randn(11, 5)

eucl = np.array([[np.sqrt(np.sum((a - b) ** 2)) for b in B] for a in A])
manh = np.array([[np.sum(np.abs(a - b)) for b in B] for a in A])

assert np.allclose(_pairwise(A, B, "euclidean"), eucl)
assert np.allclose(_pairwise(A, B, "manhattan"), manh)
print("✓ Distâncias OK\n")


# =============================================================================
# Teste 2: classificador (voto e desempate)
# =============================================================================

print("=" * 60)
print("TESTE 2: KNNClassifier (voto majoritário e desempate)")
print("=" * 60)

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([0, 0, 1, 1])

# k=2 e consulta em 1.4: vizinhos 1.0 (classe 0, d=0.4) e 2.0 (classe 1, d=0.6).
# Empate 1 a 1. Desempate: menor soma de distâncias, então vence a classe 0.
modelo = KNNClassifier(k=2, metric="euclidean").fit(X, y)
assert modelo.predict(np.array([[1.4]]))[0] == 0
assert modelo.predict(np.array([[1.6]]))[0] == 1
print("✓ Voto e desempate OK\n")


# =============================================================================
# Teste 3: regressor (média dos k vizinhos)
# =============================================================================

print("=" * 60)
print("TESTE 3: KNNRegressor (média dos k vizinhos)")
print("=" * 60)

X = np.array([[0.0], [1.0], [2.0], [10.0]])
y = np.array([0.0, 10.0, 20.0, 100.0])

# Consulta em 0.9: vizinhos 1.0 (valor 10) e 0.0 (valor 0), então a média é 5.
modelo = KNNRegressor(k=2, metric="manhattan").fit(X, y)
assert np.isclose(modelo.predict(np.array([[0.9]]))[0], 5.0)
print("✓ Média dos vizinhos OK\n")


# =============================================================================
# Teste 4: k="auto" com treino ORDENADO POR CLASSE
# (o runner estratificado entrega o treino assim; dividir em blocos em ordem
# deixaria uma classe só em cada bloco e a escolha do k daria errado)
# =============================================================================

print("=" * 60)
print('TESTE 4: k="auto" com treino ordenado por classe')
print("=" * 60)

rng = np.random.RandomState(1)
X0, X1 = rng.randn(150, 4), rng.randn(150, 4) + 3.0
X_tr, y_tr = np.vstack([X0, X1]), np.array([0] * 150 + [1] * 150)  # ordenado por classe
X_te = np.vstack([rng.randn(50, 4), rng.randn(50, 4) + 3.0])
y_te = np.array([0] * 50 + [1] * 50)

for metrica in ("euclidean", "manhattan"):
    modelo = KNNClassifier(k="auto", metric=metrica).fit(X_tr, y_tr)
    acc = np.mean(modelo.predict(X_te) == y_te)
    print(f"  {metrica}: k escolhido = {modelo.k_}, acurácia = {acc:.2f}")
    assert acc > 0.9
print("✓ k=\"auto\" OK\n")


# =============================================================================
# Teste 5: o MESMO objeto reaproveitado em vários folds (como faz o runner)
# =============================================================================

print("=" * 60)
print("TESTE 5: reuso do modelo entre folds")
print("=" * 60)

modelo = KNNClassifier(k="auto")
modelo.fit(X_tr[:200], y_tr[:200])
k1 = modelo.k_
modelo.fit(X_tr, y_tr)  # novo fit no mesmo objeto
assert modelo.X_.shape[0] == 300 and modelo.k == "auto"
print(f"  k após 1º fit = {k1}, k após 2º fit = {modelo.k_}")
print("✓ Reuso OK\n")


# =============================================================================
# Teste 6 e 7: runner do Membro 1 com os datasets reais (k fixo, para ser rápido)
# =============================================================================

print("=" * 60)
print("TESTE 6: Runner com credit-g (classificação)")
print("=" * 60)

X, y, _, class_names = to_numpy(load_arff("data/classificacao/credit-g.arff"))
print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} colunas | classes: {class_names}")
resultado = run_experiment(KNNClassifier(k=5), X, y, K=5, seed=42, task="classification")
print_tabela_classificacao({"kNN (Euclidiana, k=5)": resultado})
assert resultado["accuracy"][0] > 0.6

print("\n" + "=" * 60)
print("TESTE 7: Runner com miami_housing (regressão)")
print("=" * 60)

X, y, _, _ = to_numpy(load_arff("data/regressao/miami_housing.arff"),
                      target="SALE_PRC", ignore=["PARCELNO"])
print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} atributos")
resultado = run_experiment(KNNRegressor(k=5), X, y, K=5, seed=42, task="regression")
print_tabela_regressao({"kNN (Euclidiana, k=5)": resultado})
assert resultado["r2"][0] > 0.0  # melhor que prever a média

print("\n" + "=" * 60)
print("TODOS OS TESTES DO kNN PASSARAM!")
print("=" * 60)
