"""Testes da Regressão Linear (Membro 4). Rodar na raiz: python test_linear_regression.py"""

import numpy as np

from src.arff import load_arff, to_numpy
from src.linear_regression import LinearRegression
from src.runner import run_experiment, print_tabela_regressao


# =============================================================================
# Teste 1: recupera exatamente os coeficientes de uma relação linear perfeita
# =============================================================================

print("=" * 60)
print("TESTE 1: coeficientes de uma relação linear sem ruído")
print("=" * 60)

rng = np.random.RandomState(0)
X = rng.randn(200, 3)
# Verdade: y = 5 + 2·x0 - 3·x1 + 0.5·x2 (sem ruído).
beta_verdadeiro = np.array([5.0, 2.0, -3.0, 0.5])  # [intercepto, x0, x1, x2]
y = beta_verdadeiro[0] + X @ beta_verdadeiro[1:]

modelo = LinearRegression().fit(X, y)
assert np.isclose(modelo.intercept_, 5.0)
assert np.allclose(modelo.coef_, [2.0, -3.0, 0.5])
assert np.allclose(modelo.predict(X), y)
print(f"  intercepto = {modelo.intercept_:.4f}, coef = {modelo.coef_}")
print("✓ Coeficientes exatos OK\n")


# =============================================================================
# Teste 2: com ruído, as previsões batem com a solução via pseudo-inversa
# =============================================================================

print("=" * 60)
print("TESTE 2: concordância com a pseudo-inversa (np.linalg.pinv)")
print("=" * 60)

y_ruido = y + rng.randn(200) * 0.5

modelo = LinearRegression().fit(X, y_ruido)

# Solução de referência montando Xb e usando a pseudo-inversa diretamente.
Xb = np.hstack([np.ones((X.shape[0], 1)), X])
beta_ref = np.linalg.pinv(Xb) @ y_ruido

assert np.allclose(modelo.beta_, beta_ref)
assert np.allclose(modelo.predict(X), Xb @ beta_ref)
print("✓ Bate com a pseudo-inversa\n")


# =============================================================================
# Teste 3: matriz singular (coluna duplicada) → cai no fallback lstsq e funciona
# =============================================================================

print("=" * 60)
print("TESTE 3: atributos linearmente dependentes (XᵀX singular)")
print("=" * 60)

X_base = rng.randn(100, 2)
# Terceira coluna = soma das duas primeiras: X não tem posto cheio, logo XᵀX
# é singular e np.linalg.solve falharia. O fallback para lstsq deve resolver.
X_dep = np.hstack([X_base, (X_base[:, [0]] + X_base[:, [1]])])
y_dep = 1.0 + X_base @ np.array([2.0, -1.0])

modelo = LinearRegression().fit(X_dep, y_dep)
# As previsões ainda devem reconstruir y, mesmo com a coluna redundante.
assert np.allclose(modelo.predict(X_dep), y_dep, atol=1e-6)
print("✓ Fallback em matriz singular OK\n")


# =============================================================================
# Teste 4: runner do Membro 1 com o dataset real (miami_housing)
# =============================================================================

print("=" * 60)
print("TESTE 4: Runner com miami_housing (regressão)")
print("=" * 60)

X, y, _, _ = to_numpy(load_arff("data/regressao/miami_housing.arff"),
                      target="SALE_PRC", ignore=["PARCELNO"])
print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} atributos")
resultado = run_experiment(LinearRegression(), X, y, K=5, seed=42, task="regression")
print_tabela_regressao({"Regressão Linear Múltipla": resultado})
assert resultado["r2"][0] > 0.0   # melhor que prever a média
assert resultado["mae"][0] > 0.0

print("\n" + "=" * 60)
print("TODOS OS TESTES DA REGRESSÃO LINEAR PASSARAM!")
print("=" * 60)
