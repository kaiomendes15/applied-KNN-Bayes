# Guia de Estudo — kNN (Membro 2)

Este guia explica **tudo o que o Membro 2 implementou**, para que qualquer pessoa do grupo consiga explicar `src/knn.py` na apresentação.

| O quê | Onde |
|---|---|
| Classificador (`KNNClassifier`) e regressor (`KNNRegressor`) | `src/knn.py` |
| Distâncias, vizinhos, voto e escolha do k | `src/knn.py` (`_pairwise`, `_neighbors`, `_vote`, `select_k`) |
| Testes | `test_knn.py` |
| As 4 linhas de resultado (2 distâncias × 2 datasets) | `run_knn.py` |
| Teoria completa | `docs/fundamentacao_teorica.md`, seção 3 |

---

## 1. A ideia

O kNN **não aprende nada**. O `fit` só guarda o treino. Para prever uma amostra nova:

1. Mede a distância dela até **todas** as amostras do treino.
2. Pega as **k** mais próximas (os vizinhos).
3. Decide:
   - **classificação:** voto da maioria entre os k vizinhos;
   - **regressão:** média dos valores dos k vizinhos.

Por isso o kNN tem treino quase instantâneo e teste caro: todo o trabalho acontece no `predict`.

---

## 2. As duas distâncias

| | Fórmula | Ideia |
|---|---|---|
| **Euclidiana** | `√ Σ (xᵢ − yᵢ)²` | linha reta; diferenças grandes pesam mais (ao quadrado) |
| **Manhattan** | `Σ │xᵢ − yᵢ│` | soma das diferenças por atributo, como andar por quarteirões |

**Por que a Manhattan é bem mais lenta?** A Euclidiana usa `(a − b)² = a² + b² − 2ab`, e o termo `a @ b` é uma multiplicação de matrizes, que o numpy faz muito rápido. A Manhattan precisa do valor absoluto de cada diferença, par a par, sem esse atalho.

---

## 3. Passo a passo do código

| Função | O que faz |
|---|---|
| `_pairwise(A, B, metric)` | Matriz de distâncias (amostras de A × amostras de B), em blocos de 256 linhas para poupar memória. |
| `_neighbors(X_tr, X_te, metric, kmax)` | Para cada amostra de teste, os `kmax` vizinhos mais próximos, ordenados do mais perto ao mais longe. |
| `_vote(labels, dists, n_classes)` | Voto majoritário. **Desempate:** entre as classes empatadas, vence a de menor soma de distâncias. |
| `KNNClassifier.predict` | Vizinhos → voto. |
| `KNNRegressor.predict` | Vizinhos → média dos valores. |

**Desempate, exemplo:** com k = 2, um vizinho é `good` (a 0,4 de distância) e o outro é `bad` (a 0,6). Empate de 1 a 1; vence `good`, porque seus vizinhos estão mais perto.

---

## 4. Escolha do k (`select_k`)

Com `k="auto"`, o `fit` testa k ∈ {1, 3, 5, 7, 9, 11, 15, 21} e fica com o melhor:

1. Divide **só o treino** em 3 partes. Em cada rodada, uma parte valida e as outras duas treinam.
2. Calcula os vizinhos **uma vez** com o maior k e reaproveita os primeiros k para os valores menores.
3. Escolhe o k com a melhor média (classificação: maior acurácia; regressão: menor MAE).

O fold de teste do runner **nunca** entra nessa escolha. Se entrasse, o resultado ficaria otimista (vazamento de dados).

**Por que as linhas são embaralhadas (semente fixa) antes de dividir?** O k-fold estratificado do runner entrega o treino **ordenado por classe**. Dividir em blocos em ordem deixaria uma classe só em cada bloco, e a escolha do k daria errado. O `test_knn.py` (teste 4) cobre esse caso.

---

## 5. Como encaixa no runner

```python
modelo = KNNClassifier(k="auto", metric="manhattan")   # ou "euclidean"
modelo.fit(X_treino, y_treino)      # guarda o treino e escolhe o k (fica em modelo.k_)
y_pred = modelo.predict(X_teste)
```

- O **tempo de treino** da tabela inclui a escolha do k, porque ela acontece dentro do `fit`.
- O runner reaproveita o mesmo objeto nos 5 folds. Cada `fit` refaz tudo, então não sobra nada do fold anterior (teste 5).
- A normalização (z-score) vem do runner, e é obrigatória no kNN: sem ela, atributos de escala grande (como `credit_amount`) dominam a distância.
- O one-hot do credit-g fica **sem** `drop_first`: assim todas as categorias ficam à mesma distância umas das outras.

---

## 6. Resultados (`python run_knn.py`, 5 folds, semente 42)

**Classificação (credit-g)**

| Classificador | Acurácia | Precisão | Recall | F1-Score | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|---|
| kNN (Euclidiana) | 0.7210 ± 0.0204 | 0.6596 ± 0.0392 | 0.6055 ± 0.0251 | 0.6106 ± 0.0304 | 0.044824 ± 0.001963 | 0.006994 ± 0.001335 |
| kNN (Manhattan) | 0.7190 ± 0.0216 | 0.6650 ± 0.0576 | 0.5860 ± 0.0246 | 0.5844 ± 0.0316 | 0.204121 ± 0.009456 | 0.065813 ± 0.009804 |

**Regressão (miami_housing)**

| Regressor | MAE | R² | R² Ajustado | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|
| kNN (Euclidiana) | 55053.6198 ± 1096.7253 | 0.8704 ± 0.0182 | 0.8697 ± 0.0183 | 1.377124 ± 0.051676 | 0.495246 ± 0.041894 |
| kNN (Manhattan) | 51110.4111 ± 1179.3080 | 0.8894 ± 0.0158 | 0.8888 ± 0.0159 | 11.239665 ± 0.194474 | 4.207564 ± 0.041669 |

**O que dizer:**
- **Classificação:** as duas distâncias empatam (as diferenças ficam dentro de 1 desvio padrão), e a Euclidiana é de 5 a 9 vezes mais rápida. Recall e F1 ficam abaixo da acurácia porque `bad` é só 30% dos dados.
- **Regressão:** a Manhattan é melhor (MAE menor por mais de 3 desvios padrão, e R² maior), mas cerca de 8 vezes mais lenta. Hipótese, não testada: a Manhattan é menos sensível a valores extremos, como diz a fundamentação teórica.
- **Equilíbrio:** na classificação, a Euclidiana (mesma qualidade, bem mais rápida). Na regressão, depende do peso dado ao tempo: a Manhattan ganha em qualidade, a Euclidiana em velocidade.

> **Sobre o R² ajustado no kNN:** ele foi pensado para modelos lineares. Aqui usamos p = número de atributos para manter a comparação uniforme (ver `fundamentacao_teorica.md`).

---

## 7. Perguntas prováveis

| Pergunta | Resposta |
|---|---|
| Por que o treino é rápido e o teste é lento? | O `fit` só guarda os dados. O `predict` mede a distância até todo o treino. |
| Por que k ímpar? | Para reduzir empates na votação com duas classes. Se empatar, vale a menor soma de distâncias. |
| O k do kNN é o K do K-Fold? | Não. k é o número de vizinhos; K é o número de partições da validação cruzada. |
| Por que normalizar? | Sem z-score, atributos de escala grande dominam a distância. |
| O que é a divisão interna do treino? | Serve só para escolher o k, sem tocar no fold de teste. |
| Limitações? | Previsão cara em bases grandes, sensível à escala e à alta dimensionalidade. |
