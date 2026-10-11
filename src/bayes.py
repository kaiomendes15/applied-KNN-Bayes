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

    def __init__(self, var_smoothing=1e-2):
        # Fração da maior variância somada a todas as variâncias, para evitar divisão por zero.
        # 1e-2 em vez do usual 1e-9: colunas one-hot podem ter variância zero numa classe,
        # e um ε minúsculo pune demais quem foge dela (um fold do credit-g caiu para 0,43).
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
    """Bayesiano multivariado (normal multivariada com Σ completa por classe).

    Diferente do univariado, modela o vetor x inteiro de uma vez, então captura
    as correlações entre os atributos (fronteiras quádricas, também chamado QDA).

    Parâmetros aprendidos no `fit` (C classes, d atributos):
    - classes:     (C,)      rótulos das classes
    - prior:       (C,)      priori de cada classe
    - mean:        (C, d)    vetor de médias de cada classe
    - cov:         (C, d, d) matriz de covariância de cada classe (já regularizada)
    - inv_cov:     (C, d, d) inversa de cada Σ, pré-calculada para o predict
    - log_det_cov: (C,)      ln|Σ| de cada classe, pré-calculado para o predict
    """

    def __init__(self, reg_lambda=1e-1):
        # Valor somado à diagonal de Σ (Σ + λI) para ela sempre ter inversa.
        # Com dados em z-score (variância 1), λ = 0,1 soma 10% da variância de um
        # atributo: pouco viés, mas Σ deixa de ser quase singular nas colunas one-hot.
        self.reg_lambda = reg_lambda

    def fit(self, X, y):
        """Calcula a priori, o vetor de médias e a matriz de covariância de cada classe."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Rótulos distintos, em ordem crescente.
        self.classes = np.unique(y)
        n_classes, n_features = len(self.classes), X.shape[1]

        # Agora cada classe guarda uma matriz d × d, por isso cov tem três dimensões.
        self.prior = np.empty(n_classes)                            # (C,)
        self.mean = np.empty((n_classes, n_features))               # (C, d)
        self.cov = np.empty((n_classes, n_features, n_features))    # (C, d, d): uma Σ por classe

        # Matriz identidade d × d: uns na diagonal, zeros fora dela.
        identity = np.eye(n_features)

        for k, c in enumerate(self.classes):
            # Só as amostras (linhas) da classe c.
            Xk = X[y == c]

            # Mesmos cálculos do univariado: proporção da classe e média por coluna.
            self.prior[k] = len(Xk) / len(X)
            self.mean[k] = Xk.mean(axis=0)

            # np.cov espera variáveis nas linhas; rowvar=False avisa que estão nas colunas.
            # Diagonal = variância de cada atributo; fora dela = covariância entre pares.
            # Divide por n_k - 1 (estimador amostral, como na fundamentação teórica).
            # np.atleast_2d garante formato (d, d) mesmo quando d = 1 (np.cov devolveria escalar).
            # Soma λ só na diagonal (Σ + λI): Σ fica invertível mesmo com atributos
            # constantes ou dependentes (ex.: colunas one-hot).
            self.cov[k] = np.atleast_2d(np.cov(Xk, rowvar=False)) + self.reg_lambda * identity

        # Pré-calcula, uma vez só, as duas peças de Σ usadas no predict.
        # np.linalg.inv inverte a pilha inteira: (C, d, d) → (C, d, d).
        self.inv_cov = np.linalg.inv(self.cov)

        # ln|Σ| direto, sem calcular |Σ| antes: em d alto o determinante pode
        # estourar (overflow) ou virar zero (underflow). slogdet devolve (sinal, ln|Σ|);
        # o sinal é sempre +1 aqui, porque Σ + λI é definida positiva.
        _, self.log_det_cov = np.linalg.slogdet(self.cov)

        return self

    def predict(self, X):
        """Dá uma nota (discriminante g_k) de cada amostra para cada classe e escolhe a maior.

        g_k(x) = -½ (x - μ_k)ᵀ Σ_k⁻¹ (x - μ_k) - ½ ln|Σ_k| + ln P(ω_k)

        O termo -d/2 · ln(2π) é igual para todas as classes, então é omitido:
        não muda qual classe tem a maior nota.
        """
        X = np.asarray(X, dtype=float)

        # scores (n, C): linha = amostra, coluna = classe.
        scores = np.empty((len(X), len(self.classes)))

        # Laço nas classes (poucas); as amostras são processadas todas de uma vez.
        for k in range(len(self.classes)):
            # Passo 1: vetor de desvios de cada amostra até a média da classe → (n, d).
            diff = X - self.mean[k]

            # Passo 2: distância de Mahalanobis ao quadrado, (x - μ)ᵀ Σ⁻¹ (x - μ), por amostra.
            # diff @ inv_cov[k] faz (x - μ)ᵀ Σ⁻¹ para todas as linhas → (n, d).
            # Multiplicar elemento a elemento por diff e somar cada linha (axis=1)
            # completa o produto com (x - μ) → um número por amostra (n,).
            # Se Σ = I, vira a distância Euclidiana ao quadrado.
            mahalanobis_sq = np.sum((diff @ self.inv_cov[k]) * diff, axis=1)

            # Passo 3: monta o discriminante. Quanto mais longe (Mahalanobis)
            # e mais espalhada a classe (ln|Σ|), menor a nota; classes mais
            # frequentes (priori maior) ganham um bônus.
            scores[:, k] = (
                -0.5 * mahalanobis_sq
                - 0.5 * self.log_det_cov[k]
                + np.log(self.prior[k])
            )

        # Passo 4: posição da maior nota em cada linha → rótulo da classe.
        return self.classes[np.argmax(scores, axis=1)]
