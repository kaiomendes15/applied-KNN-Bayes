import numpy as np


def kfold_simples(X, y, K=5, seed=42):
    """K-Fold simples para regressão.
    
    Divide os dados em K partes e, em cada rodada, uma parte é teste
    e as demais são treino.
    
    Parâmetros:
        X: matriz de atributos (n, d)
        y: vetor de alvos (n,)
        K: número de folds
        seed: semente para reprodutibilidade
    
    Yields:
        (train_idx, test_idx): arrays de índices para cada fold
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    folds = np.array_split(indices, K)
    
    for i in range(K):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != i])
        yield train_idx, test_idx


def kfold_estratificado(X, y, K=5, seed=42):
    """K-Fold estratificado para classificação.
    
    Garante que cada fold mantenha a mesma proporção de classes
    do dataset original.
    
    Parâmetros:
        X: matriz de atributos (n, d)
        y: vetor de classes (n,) - inteiros
        K: número de folds
        seed: semente para reprodutibilidade
    
    Yields:
        (train_idx, test_idx): arrays de índices para cada fold
    """
    rng = np.random.default_rng(seed)
    
    # Separa índices por classe e divide cada classe em K partes
    folds_por_classe = {}
    for classe in np.unique(y):
        idx_classe = np.where(y == classe)[0]
        idx_classe = rng.permutation(idx_classe)  # embaralha dentro da classe
        folds_por_classe[classe] = np.array_split(idx_classe, K)
    
    # Monta cada fold juntando as partes de todas as classes
    for i in range(K):
        test_idx = np.concatenate([folds_por_classe[c][i] for c in folds_por_classe])
        train_idx = np.concatenate([
            folds_por_classe[c][j] 
            for c in folds_por_classe 
            for j in range(K) if j != i
        ])
        yield train_idx, test_idx
