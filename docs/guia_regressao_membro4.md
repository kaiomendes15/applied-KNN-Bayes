# Guia de Estudo — Regressão Linear, Integração e Análise (Membro 4)

Este guia explica **tudo o que o Membro 4 entregou**: a Regressão Linear Múltipla (`src/linear_regression.py`), o script que integra todos os modelos (`src/run_experiments.py`) e a análise crítica que fecha o deck. O objetivo é que qualquer pessoa do grupo consiga **explicar qualquer linha** desse código na apresentação.

**Onde está cada coisa:**

| O quê | Onde |
|---|---|
| Regressão Linear Múltipla (`LinearRegression`) | `src/linear_regression.py` |
| Script final (todos os modelos, as duas tabelas) | `src/run_experiments.py` |
| Testes | `test_linear_regression.py` |
| A 1 linha de resultado na tabela de regressão | `src/run_experiments.py` |
| Teoria completa com fórmulas | `docs/fundamentacao_teorica.md`, seção 7 |

---

## 1. A ideia

A Regressão Linear supõe que o alvo é uma **soma ponderada** dos atributos, mais um intercepto:

```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + β_d·x_d
```

- **β₀ (intercepto):** o valor de ŷ quando todos os atributos são zero.
- **β_j (coeficiente):** quanto ŷ muda quando o atributo j sobe 1 unidade e **os outros ficam parados**.

O treino procura os β que deixam as previsões o mais perto possível dos valores reais — mais precisamente, que **minimizam a soma dos quadrados dos resíduos** (os erros ŷ − y). Diferente do kNN, essa conta tem **solução direta** (fórmula fechada): não há laço de treino nem escolha de hiperparâmetro.

---

## 2. Mínimos Quadrados (OLS) e as equações normais

O erro total que queremos minimizar é a soma dos quadrados dos resíduos (SSE):

```
SSE(β) = Σ (yᵢ − ŷᵢ)²
```

Usar o **quadrado** do erro (em vez do valor absoluto) tem duas vantagens: pune erros grandes mais do que erros pequenos, e deixa a função lisa (derivável). Derivando o SSE em relação a β e igualando a zero, chega-se às **equações normais**:

```
(Xᵀ X) β = Xᵀ y
```

onde **X** já tem a coluna de 1s do intercepto. Como o SSE é uma "tigela" (função convexa), esse ponto é o **mínimo global** — a melhor reta/plano possível, sem chute inicial nem iteração.

**Interpretação geométrica:** ŷ = Xβ é a **projeção** de y no espaço gerado pelas colunas de X. O modelo escolhe, dentro de tudo que é alcançável como combinação linear dos atributos, o ponto mais perto de y.

---

## 3. A coluna de intercepto (o truque dos 1s)

Para o β₀ não ficar solto na fórmula, adicionamos uma **coluna de 1s** à esquerda de X:

```
        [ 1  x₁₁ ... x₁d ]           [ β₀ ]
   Xb = [ 1  x₂₁ ... x₂d ]      β =  [ β₁ ]
        [ ................ ]           [ ⋮  ]
        [ 1  xₙ₁ ... xₙd ]           [ β_d]
```

Como esse x₀ vale sempre 1, `β₀·1 = β₀` entra em toda previsão. Assim o intercepto vira **só mais um coeficiente**, e tudo cabe em um único produto de matrizes `ŷ = Xb · β`. É o que faz o método `_add_intercept`.

---

## 4. Passo a passo do código

```python
def fit(self, X, y):
    Xb  = self._add_intercept(X)   # coluna de 1s à esquerda → (n, d+1)
    XtX = Xb.T @ Xb                # (d+1, d+1)
    Xty = Xb.T @ y                 # (d+1,)
    self.beta_ = np.linalg.solve(XtX, Xty)   # resolve (XᵀX)β = Xᵀy
    self.intercept_, self.coef_ = self.beta_[0], self.beta_[1:]
```

| Linha | O que faz |
|---|---|
| `_add_intercept` | Prega a coluna de 1s (o x₀ do intercepto). |
| `Xb.T @ Xb` e `Xb.T @ y` | Montam os dois lados das equações normais. |
| `np.linalg.solve(XtX, Xty)` | **Resolve o sistema** sem inverter XᵀX explicitamente. |
| `beta_[0]` / `beta_[1:]` | Separam o intercepto dos coeficientes dos atributos. |

**Por que `solve` e não `inv`?** `np.linalg.inv(XtX) @ Xty` dá o mesmo resultado na teoria, mas calcular a inversa inteira é mais caro e acumula mais erro numérico. `solve` resolve o sistema diretamente (fatoração), que é o jeito recomendado.

**O fallback (`except LinAlgError → lstsq`):** se os atributos forem **linearmente dependentes** (por exemplo, colunas one-hot completas junto com o intercepto, que sempre somam 1), XᵀX fica **singular** e não tem inversa — `solve` falha. Nesse caso caímos para `np.linalg.lstsq`, que usa a **pseudo-inversa de Moore-Penrose** e devolve a solução de menor norma, que sempre existe. No miami_housing (15 atributos numéricos, ~11 mil amostras de treino por fold) isso nunca é acionado, mas deixa o modelo robusto se alguém reaproveitá-lo num dataset com one-hot.

**`predict`** é só `ŷ = Xb · β`: um produto escalar por amostra. Por isso o teste é praticamente instantâneo.

---

## 5. Custo: o oposto do kNN

| | Treino | Teste |
|---|---|---|
| **kNN** | quase nada (só guarda os dados) | caro (mede distância até todo o treino) |
| **Regressão Linear** | caro (monta e resolve o sistema, O(n·d² + d³)) | **baratíssimo** (um produto escalar, O(d) por amostra) |

Na prática, no miami_housing, a Regressão Linear treina em ~0,7 ms e prediz em ~0,1 ms por fold, enquanto o kNN leva **segundos** no teste. É a diferença entre um modelo que **resume** os dados em d+1 números (os β) e um que **carrega** todos os dados de treino.

---

## 6. Como encaixa no runner

```python
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)   # resolve as equações normais
y_pred = modelo.predict(X_teste) # ŷ = Xβ
```

- Não tem hiperparâmetro: a solução de mínimos quadrados é única.
- A normalização (z-score) do runner **não muda as previsões** do OLS, mas melhora o condicionamento numérico de XᵀX e torna os coeficientes comparáveis entre si (ver `fundamentacao_teorica.md`, 7.4).
- O **R² ajustado** usa p = número de atributos (15 no miami_housing). Ele penaliza o R² por cada atributo a mais, por isso fica levemente abaixo do R².

---

## 7. O script de integração (`src/run_experiments.py`)

É o **script final do projeto**: roda os 7 experimentos (4 de classificação + 3 de regressão) e imprime as duas tabelas. Junta as entregas de todos os membros:

```
run_experiments.py
   ├── experimentos_classificacao()  → credit-g
   │     ├── kNN (Euclidiana)        [Membro 2]
   │     ├── kNN (Manhattan)         [Membro 2]
   │     ├── Bayesiano Univariado    [Membro 3]
   │     └── Bayesiano Multivariado  [Membro 3]
   └── experimentos_regressao()      → miami_housing
         ├── kNN (Euclidiana)        [Membro 2]
         ├── kNN (Manhattan)         [Membro 2]
         └── Regressão Linear        [Membro 4]
```

**O detalhe mais importante do script:** o credit-g é carregado **duas vezes**.

- Sem `drop_first` (one-hot completo, 63 colunas): kNN e Bayesiano univariado.
- Com `drop_first=True` (50 colunas): Bayesiano multivariado, que precisa **inverter Σ** e, com as colunas one-hot completas, ela fica singular (ver `guia_bayes_membro3.md`, seção 5.6).

Rodar na raiz: `python -m src.run_experiments` (todos os números abaixo saem daí, 5 folds, seed 42).

---

## 8. Resultados completos

**Classificação (credit-g)**

| Classificador | Acurácia | Precisão | Recall | F1-Score | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|---|
| kNN (Euclidiana) | 0,721 ± 0,020 | 0,660 ± 0,039 | 0,606 ± 0,025 | 0,611 ± 0,030 | 0,036 | 0,004 |
| kNN (Manhattan) | 0,719 ± 0,022 | 0,665 ± 0,058 | 0,586 ± 0,025 | 0,584 ± 0,032 | 0,152 | 0,048 |
| Bayesiano Univariado | 0,693 ± 0,023 | 0,671 ± 0,032 | 0,698 ± 0,041 | 0,668 ± 0,029 | 0,0004 | 0,0003 |
| Bayesiano Multivariado | 0,724 ± 0,025 | 0,677 ± 0,028 | 0,681 ± 0,032 | 0,676 ± 0,028 | 0,0006 | 0,0001 |

**Regressão (miami_housing)**

| Regressor | MAE | R² | R² Ajustado | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|
| kNN (Euclidiana) | 55051 ± 1097 | 0,870 ± 0,018 | 0,870 ± 0,018 | 1,33 | 0,51 |
| kNN (Manhattan) | 51111 ± 1180 | 0,889 ± 0,016 | 0,889 ± 0,016 | 6,55 | 2,49 |
| **Regressão Linear Múltipla** | **113634 ± 2357** | **0,694 ± 0,025** | **0,693 ± 0,025** | **0,0007** | **0,0001** |

---

## 9. Análise crítica (o que vai nos slides de conclusão)

### 9.1 Desempenho × tempo

- **Regressão (o contraste mais forte):** a Regressão Linear erra quase **o dobro** do kNN (MAE ~114 mil contra ~51–55 mil) e explica bem menos da variação dos preços (R² 0,69 contra 0,87–0,89). **Mas** treina e prediz em **menos de 1 ms**, enquanto o kNN leva **segundos** por fold. É o retrato do dilema qualidade × custo.
- **Por que a Regressão Linear perde aqui?** O preço de um imóvel **não é linear** nos atributos: latitude/longitude e distâncias (oceano, centro, rodovia) têm efeito curvo e com interações. Um modelo linear só consegue traçar um plano, então sofre de **viés alto (underfitting)**. O kNN não assume nenhuma forma funcional e captura a estrutura local — por isso vai melhor exatamente onde a linear falha (ver `fundamentacao_teorica.md`, 7.5).
- **Classificação:** kNN e Bayesiano empatam em acurácia (todos em torno de 0,72), mas o Bayesiano é **centenas de vezes mais rápido** no teste (sub-milissegundo contra dezenas de milissegundos do kNN Manhattan) e tem **F1 maior** (0,67–0,68 contra 0,58–0,61), porque identifica melhor a classe minoritária `bad`.

### 9.2 Pontos fortes e limitações de cada modelo

| Modelo | Pontos fortes | Limitações |
|---|---|---|
| **Regressão Linear** | Treino e teste instantâneos; coeficientes interpretáveis (o sinal e o tamanho de cada β contam uma história); sem hiperparâmetro. | Só captura relações lineares; viés alto em dados curvos; sensível a multicolinearidade e a outliers. |
| **kNN** | Sem suposição sobre a forma dos dados; vai bem em relações não-lineares (melhor na regressão). | Teste caro em bases grandes; sensível à escala (exige z-score) e à alta dimensionalidade. |
| **Bayesiano** | Rapidíssimo; probabilístico; o multivariado aproveita correlações. | Supõe normalidade; o multivariado precisa de `drop_first` + λI para Σ não ser singular. |

### 9.3 Conclusões

1. **Não existe "melhor modelo" absoluto** — depende do critério. Na regressão, o kNN Manhattan ganha em qualidade; a Regressão Linear ganha de lavada em velocidade e interpretabilidade. Na classificação, o Bayesiano multivariado oferece o melhor equilíbrio (acurácia e F1 no topo, custo quase zero).
2. **A forma dos dados manda.** O kNN vence a regressão justamente porque o preço de imóveis é não-linear; a Regressão Linear seria competitiva num problema de fato linear.
3. **Qualidade custa tempo.** O kNN compra precisão pagando no teste; a Regressão Linear e o Bayesiano fazem o oposto. A escolha depende de quão caro é errar e de quão caro é prever.
4. **O pré-processamento foi decisivo.** z-score (kNN e condicionamento da linear), one-hot e `drop_first` (Bayesiano multivariado) mudam o resultado tanto quanto a escolha do algoritmo.

---

## 10. Perguntas prováveis na apresentação

| Pergunta | Resposta |
|---|---|
| Por que a coluna de 1s? | Para o intercepto β₀ virar só mais um coeficiente e tudo caber num produto de matrizes. |
| O que são as equações normais? | O sistema (XᵀX)β = Xᵀy que sai de minimizar a soma dos quadrados dos resíduos. Dá a solução ótima, sem iteração. |
| Por que `solve` e não `inv`? | Resolver o sistema é mais rápido e numericamente mais estável do que inverter XᵀX inteira. |
| E se XᵀX for singular? | Acontece com atributos linearmente dependentes. O código cai para `lstsq` (pseudo-inversa), que sempre dá uma solução. |
| Por que a Regressão Linear perdeu para o kNN? | Porque o preço dos imóveis não é linear nos atributos; o modelo linear tem viés alto (underfitting). |
| Por que o treino é caro e o teste é barato (ao contrário do kNN)? | O treino resolve um sistema (O(d³)); o teste é só um produto escalar (O(d) por amostra). |
| A normalização muda as previsões do OLS? | Não — só melhora o condicionamento numérico e torna os coeficientes comparáveis. |
| Usaram biblioteca de ML? | Não. Só numpy para álgebra linear (produtos de matriz, `solve`, `lstsq`). A montagem das equações normais foi feita à mão. |

---

## 11. Resumo para os slides

**Slide: Regressão Linear Múltipla**
- Modelo: ŷ = β₀ + Σ β_j x_j (soma ponderada dos atributos).
- Treino: equações normais (XᵀX)β = Xᵀy — solução fechada, sem iteração.
- Custo invertido em relação ao kNN: treino caro, teste instantâneo.

**Slide: Análise crítica (regressão)**
- kNN Manhattan: melhor qualidade (MAE 51k, R² 0,89), mas segundos por fold.
- Regressão Linear: pior qualidade (MAE 114k, R² 0,69), mas sub-milissegundo e interpretável.
- Motivo: preço de imóvel é não-linear → modelo linear faz underfitting.

**Slide: Conclusões**
- Não há melhor modelo absoluto: depende do critério (qualidade × tempo × interpretabilidade).
- A forma dos dados decide (não-linear favorece o kNN).
- O pré-processamento pesa tanto quanto o algoritmo.
