"""Métricas de avaliação para classificação e regressão."""

import numpy as np


# =============================================================================
# MÉTRICAS DE CLASSIFICAÇÃO
# =============================================================================

def accuracy(y_true, y_pred):
    """Acurácia: proporção de acertos."""
    return np.sum(y_true == y_pred) / len(y_true)


def precision_recall_f1_per_class(y_true, y_pred):
    """Calcula precisão, recall e F1 para cada classe.
    
    Retorna:
        dict com chaves sendo as classes e valores sendo (precision, recall, f1)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    resultados = {}
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        # Precisão: dos que previ como c, quantos são c?
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        # Recall: dos que são c, quantos previ como c?
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        # F1: média harmônica de precisão e recall
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        resultados[c] = (precision, recall, f1)
    
    return resultados


def precision_macro(y_true, y_pred):
    """Precisão macro: média das precisões de cada classe."""
    por_classe = precision_recall_f1_per_class(y_true, y_pred)
    return np.mean([p for p, r, f in por_classe.values()])


def recall_macro(y_true, y_pred):
    """Recall macro: média dos recalls de cada classe."""
    por_classe = precision_recall_f1_per_class(y_true, y_pred)
    return np.mean([r for p, r, f in por_classe.values()])


def f1_macro(y_true, y_pred):
    """F1 macro: média dos F1 de cada classe."""
    por_classe = precision_recall_f1_per_class(y_true, y_pred)
    return np.mean([f for p, r, f in por_classe.values()])


def classification_metrics(y_true, y_pred):
    """Retorna todas as métricas de classificação em um dict."""
    return {
        'accuracy': accuracy(y_true, y_pred),
        'precision': precision_macro(y_true, y_pred),
        'recall': recall_macro(y_true, y_pred),
        'f1': f1_macro(y_true, y_pred)
    }


# =============================================================================
# MÉTRICAS DE REGRESSÃO
# =============================================================================

def mae(y_true, y_pred):
    """MAE: Erro Absoluto Médio."""
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    """R²: Coeficiente de determinação.
    
    Mede quanto da variação do alvo o modelo explica.
    - 1.0 = perfeito
    - 0.0 = tão bom quanto prever a média
    - < 0 = pior que prever a média
    """
    ss_res = np.sum((y_true - y_pred) ** 2)      # soma dos quadrados dos resíduos
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # soma dos quadrados totais
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def r2_adjusted(y_true, y_pred, n_features):
    """R² Ajustado: penaliza pelo número de atributos.
    
    Parâmetros:
        y_true: valores reais
        y_pred: valores previstos
        n_features: número de atributos (d), sem contar o intercepto
    """
    n = len(y_true)
    r2_value = r2(y_true, y_pred)
    
    # Evita divisão por zero quando n - d - 1 <= 0
    if n - n_features - 1 <= 0:
        return r2_value
    
    return 1 - (1 - r2_value) * (n - 1) / (n - n_features - 1)


def regression_metrics(y_true, y_pred, n_features):
    """Retorna todas as métricas de regressão em um dict."""
    return {
        'mae': mae(y_true, y_pred),
        'r2': r2(y_true, y_pred),
        'r2_adj': r2_adjusted(y_true, y_pred, n_features)
    }
