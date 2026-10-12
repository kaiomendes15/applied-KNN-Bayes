"""Regressão Linear Múltipla por Mínimos Quadrados Ordinários (OLS), só com numpy.

A ideia: supor que o alvo é uma combinação linear dos atributos,
ŷ = β₀ + β₁x₁ + ... + β_d x_d, e achar os coeficientes β que minimizam a soma
dos quadrados dos resíduos. A solução tem forma fechada (equações normais),
então não há laço de treino: resolve-se um sistema linear de uma vez.

Segue o contrato do grupo (`Classe(**hiperparametros)`, `fit(X, y)`, `predict(X)`).
Diferente do kNN e do Bayes, o treino é caro (monta e resolve o sistema) e o
teste é baratíssimo (um produto escalar por amostra).
"""

import numpy as np


class LinearRegression:
    """Regressão Linear Múltipla (OLS via equações normais, com intercepto).

    Parâmetros aprendidos no `fit` (d atributos):
    - beta_:      (d+1,) vetor completo [β₀, β₁, ..., β_d], usado no predict
    - intercept_: escalar β₀ (valor de ŷ quando todos os atributos são zero)
    - coef_:      (d,)   coeficientes β₁..β_d, um por atributo

    O OLS não tem hiperparâmetros: a solução de mínimos quadrados é única.
    """

    @staticmethod
    def _add_intercept(X):
        """Adiciona uma coluna de 1s à esquerda de X (o x₀ que multiplica β₀).

        Com a coluna de 1s, o intercepto β₀ vira só mais um coeficiente, e tudo
        fica num único produto de matrizes: ŷ = X·β.
        """
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        """Resolve as equações normais (XᵀX) β = Xᵀy e guarda os coeficientes."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Xb tem a coluna de 1s do intercepto: formato (n, d+1).
        Xb = self._add_intercept(X)

        # Equações normais: derivar a soma dos quadrados dos resíduos e igualar
        # a zero leva a (XᵀX) β = Xᵀy. Como SSE é convexa, essa é a solução ótima.
        XtX = Xb.T @ Xb      # (d+1, d+1)
        Xty = Xb.T @ y       # (d+1,)

        # np.linalg.solve resolve o sistema SEM inverter XᵀX explicitamente,
        # o que é mais estável que calcular (XᵀX)⁻¹ e multiplicar.
        # Se XᵀX for singular (atributos linearmente dependentes, por exemplo
        # colunas one-hot junto com o intercepto), solve levanta LinAlgError.
        # Nesse caso caímos para lstsq, que usa a pseudo-inversa de Moore-Penrose
        # e devolve a solução de menor norma — sempre existe.
        try:
            self.beta_ = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self.beta_, *_ = np.linalg.lstsq(Xb, y, rcond=None)

        # Separa o intercepto (primeira posição) dos coeficientes dos atributos.
        self.intercept_ = self.beta_[0]
        self.coef_ = self.beta_[1:]
        return self

    def predict(self, X):
        """Previsão ŷ = X·β. Um produto escalar por amostra, por isso é rápido."""
        X = np.asarray(X, dtype=float)
        return self._add_intercept(X) @ self.beta_
