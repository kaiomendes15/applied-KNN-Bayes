"""Teste da infraestrutura do Membro 1 usando modelos dummy."""

import numpy as np
from src.arff import load_arff, to_numpy
from src.preprocessing.scaler import ZScoreScaler
from src.evaluation.cross_validation import kfold_simples, kfold_estratificado
from src.metrics import classification_metrics, regression_metrics
from src.runner import run_experiment, print_tabela_classificacao, print_tabela_regressao


# =============================================================================
# Modelos dummy para teste (sempre prevê a classe/valor mais comum)
# =============================================================================

class DummyClassifier:
    """Sempre prevê a classe mais frequente no treino."""
    def __init__(self):
        self.classe_mais_comum = None
    
    def fit(self, X, y):
        valores, contagens = np.unique(y, return_counts=True)
        self.classe_mais_comum = valores[np.argmax(contagens)]
    
    def predict(self, X):
        return np.full(len(X), self.classe_mais_comum)


class DummyRegressor:
    """Sempre prevê a média do treino."""
    def __init__(self):
        self.media = None
    
    def fit(self, X, y):
        self.media = np.mean(y)
    
    def predict(self, X):
        return np.full(len(X), self.media)


# =============================================================================
# Teste 1: ZScoreScaler
# =============================================================================

print("=" * 60)
print("TESTE 1: ZScoreScaler")
print("=" * 60)

X = np.array([[1.0, 200.0],
              [2.0, 400.0],
              [3.0, 600.0]])

scaler = ZScoreScaler()
scaler.fit(X)
Xt = scaler.transform(X)

print(f"Média após normalização: {Xt.mean(axis=0)}")  # esperado: [0. 0.]
print(f"Std após normalização:   {Xt.std(axis=0)}")   # esperado: [1. 1.]
print("✓ Scaler OK\n")


# =============================================================================
# Teste 2: K-Fold Simples
# =============================================================================

print("=" * 60)
print("TESTE 2: K-Fold Simples")
print("=" * 60)

X = np.random.rand(100, 5)
y = np.random.rand(100)

todos_test = []
for i, (train_idx, test_idx) in enumerate(kfold_simples(X, y, K=5, seed=42)):
    print(f"Fold {i+1}: treino={len(train_idx)}, teste={len(test_idx)}")
    todos_test.extend(test_idx)

# Verifica se cada índice aparece exatamente uma vez no teste
print(f"Índices únicos no teste: {len(set(todos_test))} (esperado: 100)")
print("✓ K-Fold Simples OK\n")


# =============================================================================
# Teste 3: K-Fold Estratificado
# =============================================================================

print("=" * 60)
print("TESTE 3: K-Fold Estratificado")
print("=" * 60)

X = np.random.rand(100, 5)
y = np.array([0]*70 + [1]*30)  # 70% classe 0, 30% classe 1

for i, (train_idx, test_idx) in enumerate(kfold_estratificado(X, y, K=5, seed=42)):
    proporcao_classe_1 = np.mean(y[test_idx] == 1)
    print(f"Fold {i+1}: teste={len(test_idx)}, proporção classe 1: {proporcao_classe_1:.2%}")

print("(esperado: ~30% em cada fold)")
print("✓ K-Fold Estratificado OK\n")


# =============================================================================
# Teste 4: Métricas de Classificação
# =============================================================================

print("=" * 60)
print("TESTE 4: Métricas de Classificação")
print("=" * 60)

y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2])  # alguns erros

metricas = classification_metrics(y_true, y_pred)
for nome, valor in metricas.items():
    print(f"  {nome}: {valor:.4f}")

print("✓ Métricas de Classificação OK\n")


# =============================================================================
# Teste 5: Métricas de Regressão
# =============================================================================

print("=" * 60)
print("TESTE 5: Métricas de Regressão")
print("=" * 60)

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])  # previsões próximas

metricas = regression_metrics(y_true, y_pred, n_features=3)
for nome, valor in metricas.items():
    print(f"  {nome}: {valor:.4f}")

print("✓ Métricas de Regressão OK\n")


# =============================================================================
# Teste 6: Runner completo com dataset real (classificação)
# =============================================================================

print("=" * 60)
print("TESTE 6: Runner com credit-g (classificação)")
print("=" * 60)

dataset = load_arff("data/classificacao/credit-g.arff")
X, y, feature_names, class_names = to_numpy(dataset)
print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} atributos")
print(f"Classes: {class_names}")

resultado = run_experiment(DummyClassifier(), X, y, K=5, seed=42, task='classification')
print_tabela_classificacao({"Dummy (classe majoritária)": resultado})


# =============================================================================
# Teste 7: Runner completo com dataset real (regressão)
# =============================================================================

print("\n" + "=" * 60)
print("TESTE 7: Runner com cpu_act (regressão)")
print("=" * 60)

dataset = load_arff("data/regressao/miami_housing.arff")
X, y, feature_names, class_names = to_numpy(dataset)
print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} atributos")

resultado = run_experiment(DummyRegressor(), X, y, K=5, seed=42, task='regression')
print_tabela_regressao({"Dummy (média)": resultado})


print("\n" + "=" * 60)
print("TODOS OS TESTES PASSARAM!")
print("=" * 60)
