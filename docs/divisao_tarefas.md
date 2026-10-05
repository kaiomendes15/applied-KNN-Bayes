# Divisão de Tarefas — Equipe de 4 membros

> **Atenção:** o enunciado define o projeto como **individual ou em dupla**. A formação com 4 membros precisa ser confirmada com a professora.

**Prazos:** envio dos slides até **14/10/2026** · apresentação em **15/10/2026**.

**Datasets:**
- Classificação: `credit-g` (OpenML 31).
- Regressão: `cpu_act` (OpenML 197).

**Já pronto:** leitor ARFF (`src/arff.py`), documentado em [`leitor_arff.md`](leitor_arff.md).

---

## Contrato comum

Todos os modelos seguem a mesma interface, para que possam ser plugados no runner de avaliação:

```python
modelo = Classe(**hiperparametros)
modelo.fit(X_treino, y_treino)
y_pred = modelo.predict(X_teste)
```

- `X` é uma matriz numpy `(n, d)` e `y` é um vetor numpy.
- A normalização e o one-hot ficam **fora** dos modelos (responsabilidade do Membro 1 e do leitor ARFF).
- É proibido usar scikit-learn e pandas. Só podem ser usados numpy e a biblioteca padrão do Python.

---

## Membro 1 — Infraestrutura de avaliação

**Código**
- [ ] Normalização z-score: `fit` só no treino, `transform` no treino e no teste.
- [ ] K-fold **estratificado** (classificação) e **simples** (regressão), com semente fixa.
- [ ] Métricas de classificação: Accuracy, Precision, Recall e F1 (média macro).
- [ ] Métricas de regressão: MAE, R² e R² ajustado.
- [ ] Runner: executa os 5 folds, mede o tempo de treino e de teste e calcula média ± desvio padrão.
- [ ] Geração da tabela comparativa.

**Slides**
- [ ] Banco de dados: amostras, atributos e o que cada dataset representa.
- [ ] Métricas de avaliação.

> É o caminho crítico do projeto. Os demais membros dependem do runner para obter os resultados finais.

---

## Membro 2 — kNN (classificação e regressão)

**Código**
- [ ] Distâncias Euclidiana e Manhattan.
- [ ] kNN classificador (voto majoritário com regra de desempate).
- [ ] kNN regressor (média dos k vizinhos).
- [ ] Escolha de k com validação interna ao treino.

**Experimentos:** 2 linhas na tabela de classificação e 2 na de regressão.

**Slides**
- [ ] Funcionamento do kNN e das duas distâncias.
- [ ] Comparação Euclidiana × Manhattan.

---

## Membro 3 — Classificadores Bayesianos

**Código**
- [ ] **Univariado:** μ e σ² por classe e por atributo, discriminante em escala log e *variance smoothing*.
- [ ] **Multivariado:** μ e Σ por classe, log-determinante, inversa de Σ e regularização Σ + λI.
- [ ] Usar `drop_first=True` no credit-g para evitar Σ singular.

**Experimentos:** 2 linhas na tabela de classificação.

**Slides**
- [ ] Teorema de Bayes e regra MAP.
- [ ] Caso univariado × multivariado.

---

## Membro 4 — Regressão Linear, integração e apresentação

**Código**
- [ ] Regressão Linear Múltipla: mínimos quadrados via equações normais, com coluna de intercepto.
- [ ] Script final que roda todos os modelos nos dois datasets e gera as tabelas.

**Experimentos:** 1 linha na tabela de regressão.

**Slides**
- [ ] Montagem e padronização visual do deck.
- [ ] Introdução e justificativa dos datasets.
- [ ] Análise crítica: desempenho × tempo, pontos fortes e limitações.
- [ ] Conclusões e referências.
- [ ] Envio dos slides no AVA.

---

## Cronograma

| Período | Atividade |
|---|---|
| **05–07/10** | M1: normalização, k-fold e métricas. M2, M3 e M4: implementação dos modelos, testando com um split simples. |
| **08/10** | M1 entrega o runner. Integração de todos os modelos. |
| **09–10/10** | Experimentos completos e geração das tabelas. Ajustes (Σ singular, escolha de k). |
| **11–12/10** | Cada membro escreve seus slides. M4 consolida o deck e a análise crítica. |
| **13/10** | Revisão geral e ensaio da apresentação. |
| **14/10** | **Envio dos slides no AVA.** |

---

## Organização no Git

**Branches**

| Membro | Branch |
|---|---|
| M1 | `feat/avaliacao` |
| M2 | `feat/knn` |
| M3 | `feat/bayes` |
| M4 | `feat/regressao-linear` |

**Integração:** Pull Request para a `main`.

**Commits:** padrão Conventional Commits (`feat:`, `fix:`, `docs:`…).

**Estrutura de arquivos**

```
src/
├── arff.py              ✅ pronto
├── preprocessing.py     M1
├── cross_validation.py  M1
├── metrics.py           M1
├── knn.py               M2
├── bayes.py             M3
├── linear_regression.py M4
└── run_experiments.py   M4
```

---

## Resultados esperados

**Classificação (credit-g)**

| Classificador | Acurácia | Precisão | Recall | F1-Score | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|---|
| kNN (Euclidiana) | | | | | | |
| kNN (Manhattan) | | | | | | |
| Bayesiano Univariado | | | | | | |
| Bayesiano Multivariado | | | | | | |

**Regressão (cpu_act)**

| Regressor | MAE | R² | R² Ajustado | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|
| kNN (Euclidiana) | | | | | |
| kNN (Manhattan) | | | | | |
| Regressão Linear Múltipla | | | | | |

Todos os valores devem ser apresentados no formato **média ± desvio padrão** entre os 5 folds.

---

> Todos devem conseguir explicar qualquer parte do código na apresentação, já que o projeto exige que os algoritmos sejam implementados manualmente.
