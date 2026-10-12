"""k-Nearest Neighbors (kNN) implementado apenas com numpy.

Classes:
- `KNNClassifier`: classificação por voto majoritário entre os k vizinhos.
- `KNNRegressor`: regressão pela média dos valores dos k vizinhos.

Ambas seguem o contrato do grupo (`Classe(**hiperparametros)`, `fit(X, y)`, `predict(X)`)
e aceitam duas distâncias: "euclidean" e "manhattan".

O kNN não aprende parâmetros: o `fit` só guarda o treino. Todo o custo vem no `predict`,
que mede a distância de cada amostra de teste até todo o treino.

Com `k="auto"`, o próprio `fit` escolhe o k por validação interna, usando só o treino.
"""

import numpy as np

METRICS = ("euclidean", "manhattan")


def _pairwise(A, B, metric, chunk=256):
    """Matriz de distâncias (len(A), len(B)): linha = amostra de A, coluna = amostra de B.

    Calculada em blocos de `chunk` linhas para não estourar a memória.
    """
    out = np.empty((A.shape[0], B.shape[0]))

    if metric == "euclidean":
        # Usa (a - b)² = a² + b² - 2ab: a parte "a @ b" é uma multiplicação de matrizes,
        # que o numpy faz muito rápido. É por isso que a Euclidiana é bem mais veloz.
        b2 = (B ** 2).sum(axis=1)
        for i in range(0, A.shape[0], chunk):
            a = A[i:i + chunk]
            d2 = (a ** 2).sum(axis=1)[:, None] + b2[None, :] - 2.0 * a @ B.T
            # maximum(…, 0): erros de arredondamento podem dar -1e-13, e a raiz quebraria.
            out[i:i + chunk] = np.sqrt(np.maximum(d2, 0.0))

    elif metric == "manhattan":
        # Soma de |a - b| por atributo. Não tem atalho de multiplicação de matrizes:
        # o numpy compara cada par (amostra de A, amostra de B), por isso é mais lento.
        for i in range(0, A.shape[0], chunk):
            a = A[i:i + chunk]
            out[i:i + chunk] = np.abs(a[:, None, :] - B[None, :, :]).sum(axis=2)

    else:
        raise ValueError(f"metric deve ser um de {METRICS}")
    return out


def _neighbors(X_tr, X_te, metric, kmax):
    """Índices e distâncias dos `kmax` vizinhos mais próximos de cada amostra de teste.

    Devolve dois arrays (n_teste, kmax), do vizinho mais perto ao mais longe.
    """
    D = _pairwise(X_te, X_tr, metric)
    kmax = min(kmax, X_tr.shape[0])
    # argpartition separa os kmax menores sem ordenar tudo (mais barato que argsort completo).
    part = np.argpartition(D, kmax - 1, axis=1)[:, :kmax]
    dist = np.take_along_axis(D, part, axis=1)
    # Agora ordena só esses kmax, para que o corte [:k] pegue os k mais próximos.
    order = np.argsort(dist, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(dist, order, axis=1)


def _vote(labels, dists, n_classes):
    """Voto majoritário entre os vizinhos de cada amostra.

    Regra de desempate: entre as classes empatadas, vence a que tem a menor soma de
    distâncias (ou seja, a que tem os vizinhos mais próximos).
    """
    pred = np.empty(labels.shape[0], dtype=int)
    for i in range(labels.shape[0]):
        counts = np.bincount(labels[i], minlength=n_classes)
        tied = np.flatnonzero(counts == counts.max())
        if len(tied) == 1:
            pred[i] = tied[0]
        else:
            sums = [dists[i][labels[i] == c].sum() for c in tied]
            pred[i] = tied[int(np.argmin(sums))]
    return pred


class _KNN:
    """Base comum: guarda o treino, escolhe o k e acha os vizinhos."""

    def __init__(self, k=5, metric="euclidean"):
        # k: número de vizinhos (int), ou "auto" para escolher por validação interna.
        if metric not in METRICS:
            raise ValueError(f"metric deve ser um de {METRICS}")
        self.k, self.metric = k, metric

    def fit(self, X, y):
        """Guarda o treino. Com k="auto", também escolhe o k (usando só este treino)."""
        self.X_ = np.asarray(X, dtype=float)
        self._set_y(np.asarray(y))
        self.k_ = select_k(self.X_, np.asarray(y), self.metric, self._task) if self.k == "auto" else self.k
        return self

    def predict(self, X):
        """Para cada amostra de X: acha os k_ vizinhos no treino e decide."""
        idx, dist = _neighbors(self.X_, np.asarray(X, dtype=float), self.metric, self.k_)
        return self._predict_from(idx[:, :self.k_], dist[:, :self.k_])


class KNNClassifier(_KNN):
    """kNN para classificação (voto majoritário, desempate pela menor distância total)."""

    _task = "classification"

    def _set_y(self, y):
        # y_ guarda a classe como índice 0..C-1; classes_ traduz de volta para o rótulo original.
        self.classes_, self.y_ = np.unique(y, return_inverse=True)

    def _predict_from(self, idx, dist):
        pred = _vote(self.y_[idx], dist, len(self.classes_))
        return self.classes_[pred]


class KNNRegressor(_KNN):
    """kNN para regressão (média dos valores dos k vizinhos)."""

    _task = "regression"

    def _set_y(self, y):
        self.y_ = y.astype(float)

    def _predict_from(self, idx, dist):
        return self.y_[idx].mean(axis=1)


def select_k(X, y, metric="euclidean", task="classification",
             ks=(1, 3, 5, 7, 9, 11, 15, 21), n_splits=3, seed=0):
    """Escolhe o k por validação interna, usando SÓ os dados de treino recebidos.

    Divide o treino em `n_splits` partes; em cada rodada, uma parte valida e as outras
    treinam. O k com o melhor resultado médio vence (classificação: maior acurácia;
    regressão: menor MAE). O fold de teste do runner nunca é tocado aqui.

    As linhas são embaralhadas (semente fixa) antes de dividir. Isso é necessário porque
    o treino pode chegar ordenado, por exemplo por classe, e blocos em ordem ficariam
    com uma classe só.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    # Descarta k grande demais para o tamanho do treino interno.
    ks = [k for k in ks if k < len(X) * (n_splits - 1) / n_splits]
    folds = np.array_split(np.random.RandomState(seed).permutation(len(X)), n_splits)
    Model = KNNClassifier if task == "classification" else KNNRegressor

    scores = {k: [] for k in ks}
    for i in range(n_splits):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        m = Model(k=max(ks), metric=metric).fit(X[tr], y[tr])
        # Calcula os vizinhos UMA vez com o maior k e reaproveita o prefixo para os menores.
        idx, dist = _neighbors(m.X_, X[va], metric, max(ks))
        for k in ks:
            p = m._predict_from(idx[:, :k], dist[:, :k])
            scores[k].append((p == y[va]).mean() if task == "classification"
                             else -np.abs(p - y[va]).mean())
    return max(ks, key=lambda k: np.mean(scores[k]))
