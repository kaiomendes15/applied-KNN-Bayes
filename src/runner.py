"""Runner de experimentos: executa k-fold, mede tempos e calcula métricas."""

import time
import numpy as np

from src.preprocessing.scaler import ZScoreScaler
from src.evaluation.cross_validation import kfold_simples, kfold_estratificado
from src.metrics import classification_metrics, regression_metrics


def run_experiment(modelo, X, y, K=5, seed=42, task='classification'):
    """Executa validação cruzada k-fold e retorna métricas com média ± std.
    
    Parâmetros:
        modelo: objeto com métodos fit(X, y) e predict(X)
        X: matriz de atributos (n, d)
        y: vetor de alvos (n,)
        K: número de folds
        seed: semente para reprodutibilidade
        task: 'classification' ou 'regression'
    
    Retorna:
        dict com {métrica: (média, std)} para cada métrica + tempos
    """
    # Escolhe o tipo de k-fold
    if task == 'classification':
        kfold = kfold_estratificado(X, y, K, seed)
    else:
        kfold = kfold_simples(X, y, K, seed)
    
    resultados = []
    n_features = X.shape[1]
    
    for train_idx, test_idx in kfold:
        # Separa treino e teste
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # Normaliza (fit só no treino)
        scaler = ZScoreScaler()
        scaler.fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # Treina e mede tempo
        t0 = time.perf_counter()
        modelo.fit(X_tr, y_tr)
        tempo_treino = time.perf_counter() - t0
        
        # Prediz e mede tempo
        t0 = time.perf_counter()
        y_pred = modelo.predict(X_te)
        tempo_teste = time.perf_counter() - t0
        
        # Calcula métricas
        if task == 'classification':
            metricas = classification_metrics(y_te, y_pred)
        else:
            metricas = regression_metrics(y_te, y_pred, n_features)
        
        metricas['tempo_treino'] = tempo_treino
        metricas['tempo_teste'] = tempo_teste
        resultados.append(metricas)
    
    return _resumir(resultados)


def _resumir(resultados):
    """Calcula média e desvio padrão de cada métrica entre os folds."""
    chaves = resultados[0].keys()
    return {
        k: (
            np.mean([r[k] for r in resultados]),
            np.std([r[k] for r in resultados], ddof=1)
        )
        for k in chaves
    }


def formatar_resultado(media, std, decimais=4):
    """Formata um resultado como 'média ± std'."""
    return f"{media:.{decimais}f} ± {std:.{decimais}f}"


def gerar_tabela_classificacao(resultados_modelos):
    """Gera tabela de resultados para classificação.
    
    Parâmetros:
        resultados_modelos: dict {nome_modelo: resultado_do_run_experiment}
    
    Retorna:
        string com a tabela formatada em markdown
    """
    linhas = []
    linhas.append("| Classificador | Acurácia | Precisão | Recall | F1-Score | Tempo Treino (s) | Tempo Teste (s) |")
    linhas.append("|---|---|---|---|---|---|---|")
    
    for nome, res in resultados_modelos.items():
        linha = f"| {nome} "
        linha += f"| {formatar_resultado(*res['accuracy'])} "
        linha += f"| {formatar_resultado(*res['precision'])} "
        linha += f"| {formatar_resultado(*res['recall'])} "
        linha += f"| {formatar_resultado(*res['f1'])} "
        linha += f"| {formatar_resultado(*res['tempo_treino'], decimais=6)} "
        linha += f"| {formatar_resultado(*res['tempo_teste'], decimais=6)} |"
        linhas.append(linha)
    
    return "\n".join(linhas)


def gerar_tabela_regressao(resultados_modelos):
    """Gera tabela de resultados para regressão.
    
    Parâmetros:
        resultados_modelos: dict {nome_modelo: resultado_do_run_experiment}
    
    Retorna:
        string com a tabela formatada em markdown
    """
    linhas = []
    linhas.append("| Regressor | MAE | R² | R² Ajustado | Tempo Treino (s) | Tempo Teste (s) |")
    linhas.append("|---|---|---|---|---|---|")
    
    for nome, res in resultados_modelos.items():
        linha = f"| {nome} "
        linha += f"| {formatar_resultado(*res['mae'])} "
        linha += f"| {formatar_resultado(*res['r2'])} "
        linha += f"| {formatar_resultado(*res['r2_adj'])} "
        linha += f"| {formatar_resultado(*res['tempo_treino'], decimais=6)} "
        linha += f"| {formatar_resultado(*res['tempo_teste'], decimais=6)} |"
        linhas.append(linha)
    
    return "\n".join(linhas)


def print_tabela_classificacao(resultados_modelos):
    """Imprime a tabela de classificação no console."""
    print("\n=== RESULTADOS - CLASSIFICAÇÃO ===\n")
    print(gerar_tabela_classificacao(resultados_modelos))


def print_tabela_regressao(resultados_modelos):
    """Imprime a tabela de regressão no console."""
    print("\n=== RESULTADOS - REGRESSÃO ===\n")
    print(gerar_tabela_regressao(resultados_modelos))
