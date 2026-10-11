# Verossimilhança no Naive Bayes — Categórica × Gaussiana

Este guia complementa o [`naive_bayes.md`](naive_bayes.md). Ele explica as **duas formas de calcular a verossimilhança**, que é a parte do Naive Bayes que mede "quão típico este valor é dentro desta classe". Os exemplos usam números reais do `credit-g`.

---

## 1. Onde a verossimilhança entra

Relembrando a regra do Naive Bayes:

```
pontuação(classe) = priori(classe) × p(atributo 1 | classe) × p(atributo 2 | classe) × ...
                                     └──────────────── verossimilhanças ────────────────┘
```

A **priori** é sempre calculada do mesmo jeito: fração da classe no treino.

A **verossimilhança** de cada atributo depende do **tipo de valor** que ele guarda:

| Tipo do atributo | Que valores ele tem? | Abordagem |
|---|---|---|
| **Categórico** | Rótulos sem conta possível: `rent`, `own`, `for free` | **Contagem** (frequência) |
| **Numérico** | Números contínuos: 12, 36, 48 meses | **Gaussiana** (média e variância) |

> **Regra prática:** se faz sentido calcular a média do atributo, ele é numérico. "Média do tipo de moradia" não faz sentido, então `housing` é categórico.

---

## 2. Abordagem categórica: contagem

### A ideia

Basta **contar**. Entre os clientes de uma classe, que fração tem aquele valor?

### A fórmula

$$
P(x_j = v \mid \omega_k) = \frac{\text{nº de amostras da classe } k \text{ com o valor } v}{\text{nº de amostras da classe } k}
$$

### Exemplo: atributo `housing`

| Classe | `rent` | `own` | `for free` | Total |
|---|---|---|---|---|
| `good` | 109 | 527 | 64 | 700 |
| `bad` | 70 | 186 | 44 | 300 |

Um cliente novo mora de aluguel (`rent`):

- P(rent | good) = 109 / 700 = **0,156**
- P(rent | bad) = 70 / 300 = **0,233**

Morar de aluguel é proporcionalmente mais comum entre os maus pagadores, então esse atributo "puxa" a decisão para `bad`.

### O problema do zero e a suavização de Laplace

No credit-g, ninguém tem `purpose = vacation`. A contagem dá **0 / 700 = 0**. Como a pontuação é uma **multiplicação**, um único zero anula tudo: a classe perde, não importa o que digam os outros atributos.

A solução é a **suavização de Laplace**: somar 1 em cada contagem.

$$
P(x_j = v \mid \omega_k) = \frac{\text{contagem} + 1}{n_k + \text{nº de categorias do atributo}}
$$

`purpose` tem 11 categorias, então:

- P(vacation | good) = (0 + 1) / (700 + 11) = **0,0014**, uma probabilidade pequena, mas diferente de zero.

Somamos o número de categorias no denominador para que as probabilidades de todas as categorias continuem somando 1.

### Quando usar

- Atributos com **poucas categorias fixas**, como tipo de moradia, finalidade do empréstimo ou estado civil.
- Atributos de "sim/não", como `own_telephone`.

---

## 3. Abordagem gaussiana: média e variância

### A ideia

Números contínuos raramente se repetem exatamente. Contar quantos clientes têm **exatamente** 4.217 de crédito não ajuda. Por isso, resumimos cada classe com uma **curva em forma de sino** e perguntamos: quão perto do centro do sino está o valor do cliente?

### As fórmulas

No treino, calculamos para cada classe e cada atributo:

$$
\mu_{kj} = \text{média do atributo } j \text{ na classe } k \qquad \sigma^2_{kj} = \text{variância do atributo } j \text{ na classe } k
$$

Na previsão, usamos a densidade da normal (fórmula do enunciado):

$$
p(x_j \mid \omega_k) = \frac{1}{\sqrt{2\pi\sigma^2_{kj}}} \exp\left[-\frac{(x_j - \mu_{kj})^2}{2\sigma^2_{kj}}\right]
$$

Lendo a fórmula em partes:

- `(x − μ)²`: quão longe o valor está do centro. Quanto mais longe, menor o resultado.
- `σ²`: a largura do sino. Numa classe com valores muito espalhados, ficar longe da média "pesa" menos.

### Exemplo: atributo `duration` (meses)

| Classe | Média (μ) | Desvio (σ) |
|---|---|---|
| `good` | 19 | 11 |
| `bad` | 25 | 13 |

Um cliente novo pede **48 meses**:

- p(48 | good) = **0,00112**, porque 48 está a mais de 2,5 desvios da média dos bons.
- p(48 | bad) = **0,00642**, porque 48 está mais perto do perfil dos maus pagadores.

> O resultado é uma **densidade**, não uma probabilidade. Ele pode até passar de 1. Isso não é problema, porque só usamos o valor para **comparar** classes.

### O problema da variância zero

Se um atributo tem o mesmo valor em todas as amostras de uma classe, σ² = 0 e a fórmula **divide por zero**. É o equivalente do "problema do zero" da abordagem categórica.

A solução é a **suavização de variância** (`var_smoothing`): somar um valor pequeno a σ² e usar σ² + ε.

### Quando usar

- Atributos **numéricos contínuos**, como duração, valor do crédito ou idade.
- Funciona melhor quando o histograma do atributo dentro de cada classe tem formato parecido com um sino.

---

## 4. Comparação lado a lado

| | **Categórica (contagem)** | **Gaussiana** |
|---|---|---|
| Tipo de atributo | Categorias (`rent`, `own`…) | Números contínuos |
| O que o `fit` guarda | Tabela de contagens por classe | Média e variância por classe |
| Cálculo na previsão | Consulta a tabela | Aplica a fórmula da normal |
| Resultado | Probabilidade (0 a 1) | Densidade (pode passar de 1) |
| Problema do zero | Categoria nunca vista → **Laplace** | Variância zero → **var_smoothing** |
| Suposição | Nenhuma sobre a forma dos dados | Dados em formato de sino |

---

## 5. Juntando as duas: Naive Bayes misto

Como a regra é uma multiplicação, cada atributo pode usar a sua própria abordagem. **Cliente novo:** mora de aluguel e pede 48 meses.

| Classe | Priori | P(rent \| classe) — contagem | p(48 \| classe) — gaussiana | Pontuação |
|---|---|---|---|---|
| `good` | 0,70 | 0,156 | 0,00112 | 0,70 × 0,156 × 0,00112 = **1,2 × 10⁻⁴** |
| `bad` | 0,30 | 0,233 | 0,00642 | 0,30 × 0,233 × 0,00642 = **4,5 × 10⁻⁴** |

**Resultado: `bad`.** Os dois atributos apontaram para `bad` e superaram a priori, que favorecia `good`.

### Com logaritmo (como no código)

Trocamos o produto por uma soma de logs para evitar números minúsculos:

| Classe | log(priori) + log(contagem) + log(gaussiana) | Log-pontuação |
|---|---|---|
| `good` | log(0,70) + log(0,156) + log(0,00112) | **−9,01** |
| `bad` | log(0,30) + log(0,233) + log(0,00642) | **−7,71** ← maior |

A classe vencedora é a mesma. O log nunca muda a decisão.

---

## 6. Estrutura do código

### Visão geral

```
fit(X, y)
 ├── priori de cada classe
 ├── atributos categóricos → tabelas de contagem (com Laplace)
 └── atributos numéricos   → média e variância (com var_smoothing)

predict(X)
 └── para cada amostra e cada classe:
       log(priori)
     + soma dos log das verossimilhanças categóricas  (consulta a tabela)
     + soma dos log das verossimilhanças gaussianas   (aplica a fórmula)
     → escolhe a classe de maior soma
```

### `fit` — pseudocódigo

```
classes = valores únicos de y

para cada classe k:
    Xk = linhas da classe k
    priori[k] = len(Xk) / len(X)

    # gaussiana: um número por atributo numérico
    media[k]     = média de cada coluna numérica de Xk
    variancia[k] = variância de cada coluna numérica de Xk + ε

    # categórica: uma tabela por atributo categórico
    para cada atributo categórico j:
        para cada categoria v de j:
            prob[k][j][v] = (contagem de v em Xk + 1) / (len(Xk) + nº de categorias de j)
```

### `predict` — pseudocódigo

```
para cada amostra x:
    para cada classe k:
        pontuacao[k] = log(priori[k])

        para cada atributo numérico j:
            pontuacao[k] += -0.5 * ( log(2π · variancia[k][j]) + (x[j] - media[k][j])² / variancia[k][j] )

        para cada atributo categórico j:
            pontuacao[k] += log( prob[k][j][ x[j] ] )

    previsão = classe com a maior pontuacao
```

---

## 7. E no nosso projeto?

O enunciado pede o **Bayesiano Gaussiano univariado**, ou seja, a abordagem da **Seção 3** para todos os atributos. Por isso:

1. O leitor ARFF transforma os 13 atributos categóricos do credit-g em colunas **one-hot** (0 ou 1).
2. O classificador trata **todas as colunas como numéricas**, com média e variância.

O efeito disso:

- **Funciona:** a média de uma coluna 0/1 é exatamente a fração de amostras com aquela categoria, então a informação da contagem está lá.
- **É uma aproximação:** uma coluna que só vale 0 ou 1 não tem formato de sino, então a gaussiana não descreve bem esses dados.
- **A abordagem categórica seria a mais adequada** para esses 13 atributos. Esse é um bom ponto para a seção de **limitações** dos slides.

---

## 8. Resumo para os slides

| Pergunta | Resposta curta |
|---|---|
| O que é verossimilhança? | Quão típico o valor do atributo é dentro de uma classe. |
| Como calcular para categorias? | Contagem: fração da classe com aquele valor. |
| Como calcular para números? | Gaussiana: densidade da normal com a média e a variância da classe. |
| O que fazer com zeros? | Laplace (categórica) e `var_smoothing` (gaussiana). |
| Dá para misturar? | Sim: cada atributo usa sua abordagem e tudo é multiplicado (ou somado em log). |
| O que usamos no projeto? | Gaussiana em tudo, com os categóricos em one-hot. É uma aproximação e uma limitação a citar. |
