"""Classificadores Bayesianos Gaussianos implementados apenas com numpy.

1. `GaussianNaiveBayes` (caso univariado): supõe atributos independentes dentro
   de cada classe, com uma normal univariada por atributo.
2. `GaussianBayes` (caso multivariado): uma normal multivariada por classe,
   com matriz de covariância completa.

Ambos seguem o contrato `fit(X, y)` / `predict(X)` e decidem pela regra MAP,
calculada em escala logarítmica.
"""

import numpy as np


class GaussianNaiveBayes:
    """Bayesiano univariado (Naive Bayes Gaussiano).

    Parâmetros aprendidos no `fit` (C classes, d atributos):
    - classes: (C,)   rótulos das classes
    - prior:   (C,)   priori de cada classe
    - mean:    (C, d) média de cada atributo, por classe
    - var:     (C, d) variância de cada atributo, por classe (já suavizada)
    """

    def __init__(self, var_smoothing=1e-9):
        # Fração da maior variância somada a todas as variâncias, para evitar divisão por zero.
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """Calcula a priori, as médias e as variâncias de cada classe."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Rótulos distintos, em ordem crescente.
        self.classes = np.unique(y)
        n_classes, n_features = len(self.classes), X.shape[1]

        # np.empty reserva o formato sem inicializar (valores lixo); o laço sobrescreve tudo.
        self.prior = np.empty(n_classes)                # (C,): uma posição por classe
        self.mean = np.empty((n_classes, n_features))   # (C, d): linha = classe, coluna = atributo
        self.var = np.empty((n_classes, n_features))    # (C, d): mesmo formato de mean

        # k = posição (linha dos arrays); c = rótulo (usado no filtro).
        # Separados porque os rótulos nem sempre são 0, 1, 2...
        for k, c in enumerate(self.classes):
            # y == c gera True/False por amostra; X[máscara] mantém só as linhas True.
            Xk = X[y == c]

            # len conta linhas: amostras da classe / total de amostras.
            self.prior[k] = len(Xk) / len(X)

            # axis=0 percorre as linhas: um valor por coluna (atributo).
            self.mean[k] = Xk.mean(axis=0)

            # .var divide por n (não n - 1): variância de máxima verossimilhança.
            self.var[k] = Xk.var(axis=0)

        # Soma um ε a todas as variâncias para nenhuma ficar zero.
        # ε é proporcional à maior variância de X, então se adapta à escala dos dados.
        self.epsilon = self.var_smoothing * X.var(axis=0).max()
        self.var += self.epsilon

        return self

    def predict(self, X):
        """Dá uma nota de cada amostra para cada classe e escolhe a maior.

        Tudo é calculado de uma vez, numa "tabela" amostra × classe × atributo,
        com formato (n, C, d).
        """
        X = np.asarray(X, dtype=float)

        # Passo 1: distância de cada valor até a média da classe.
        # np.newaxis abre espaço para comparar cada amostra com cada classe:
        # X (n, 1, d) - mean (C, d) → diff (n, C, d).
        diff = X[:, np.newaxis, :] - self.mean

        # Passo 2: transforma cada distância em nota (log da gaussiana).
        # A nota é negativa: quanto mais longe da média, mais negativa.
        log_likelihood = -0.5 * (np.log(2 * np.pi * self.var) + diff**2 / self.var)

        # Passos 3 e 4: soma as notas de todos os atributos (axis=2)
        # e depois soma o log da priori de cada classe → scores (n, C).
        scores = np.log(self.prior) + log_likelihood.sum(axis=2)

        # Passo 5: argmax diz em qual coluna (classe) está a maior nota de cada linha;
        # self.classes troca essa posição pelo rótulo da classe.
        return self.classes[np.argmax(scores, axis=1)]


class GaussianBayes:
    """Bayesiano multivariado (normal multivariada com Σ completa por classe)."""

    def __init__(self, reg_lambda=1e-6):
        self.reg_lambda = reg_lambda

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
