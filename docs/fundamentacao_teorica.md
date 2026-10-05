# Fundamentação Teórica — Projeto AV2 de Inteligência Artificial Computacional

Este documento reúne a teoria necessária para desenvolver o projeto: os algoritmos de classificação e regressão, o protocolo de validação cruzada e as métricas de avaliação. O foco é o **conceito** e a **matemática** de cada etapa, sem código.

---

## Sumário

1. [Aprendizado supervisionado: conceitos e notação](#1-aprendizado-supervisionado-conceitos-e-notação)
2. [Pré-processamento dos dados](#2-pré-processamento-dos-dados)
3. [k-Nearest Neighbors (kNN)](#3-k-nearest-neighbors-knn)
4. [Teoria de Decisão Bayesiana](#4-teoria-de-decisão-bayesiana)
5. [Classificador Bayesiano — Caso Univariado](#5-classificador-bayesiano--caso-univariado)
6. [Classificador Bayesiano — Caso Multivariado](#6-classificador-bayesiano--caso-multivariado)
7. [Regressão Linear Múltipla](#7-regressão-linear-múltipla)
8. [Validação Cruzada k-Fold](#8-validação-cruzada-k-fold)
9. [Métricas de Classificação](#9-métricas-de-classificação)
10. [Métricas de Regressão](#10-métricas-de-regressão)
11. [Comparação entre os algoritmos](#11-comparação-entre-os-algoritmos)
12. [Referências](#12-referências)

---

## 1. Aprendizado supervisionado: conceitos e notação

No **aprendizado supervisionado**, temos um conjunto de exemplos rotulados e queremos aprender uma função que mapeie as entradas para as saídas, de modo a prever corretamente a saída de exemplos novos.

- **Classificação:** a saída é uma categoria discreta, por exemplo "spam" ou "não spam", ou as classes $\{\omega_1, \dots, \omega_C\}$.
- **Regressão:** a saída é um valor numérico contínuo, por exemplo o preço de um imóvel.

### Notação usada no documento

| Símbolo | Significado |
|---|---|
| $n$ | número de amostras (instâncias) |
| $d$ (ou $p$) | número de atributos (variáveis preditoras) |
| $\mathbf{x} = (x_1, \dots, x_d)^t$ | vetor de atributos de uma amostra (vetor coluna) |
| $\mathbf{X}$ | matriz de dados $n \times d$, em que cada linha é uma amostra |
| $y$ | rótulo (classe) ou valor-alvo (regressão) |
| $\hat{y}$ | valor previsto pelo modelo |
| $\omega_k$ | a $k$-ésima classe |
| $C$ | número de classes |

### Treino e teste

Todo modelo passa por duas fases:

- **Treino (fit):** os parâmetros são estimados a partir dos dados de treino.
- **Teste (predict):** o modelo treinado faz previsões para dados que **não viu** durante o treino.

O objetivo é a **generalização**, ou seja, ter bom desempenho em dados novos e não apenas nos dados de treino. Um modelo que decora o treino e erra nos dados novos sofre de **overfitting**. Um modelo simples demais para capturar o padrão sofre de **underfitting**.

---

## 2. Pré-processamento dos dados

Antes de treinar, os dados costumam precisar de preparação. Os passos abaixo são especialmente importantes para os algoritmos deste projeto.

### 2.1 Normalização (escalonamento) dos atributos

O kNN usa distâncias. Por isso, atributos com escalas grandes (por exemplo, salário em reais) **dominam** atributos com escalas pequenas (por exemplo, idade em anos). As duas formas mais comuns de normalizar são:

**Padronização (z-score):** cada atributo passa a ter média 0 e desvio padrão 1.

$$
x'_j = \frac{x_j - \mu_j}{\sigma_j}
$$

**Min-Max:** cada atributo passa a ficar no intervalo $[0, 1]$.

$$
x'_j = \frac{x_j - \min_j}{\max_j - \min_j}
$$

> **Regra fundamental (evitar vazamento de dados):** os parâmetros da normalização ($\mu_j$, $\sigma_j$, $\min_j$, $\max_j$) devem ser calculados **apenas com os dados de treino** de cada fold e depois aplicados aos dados de teste. Calcular esses valores com o dataset inteiro deixa informação do teste "vazar" para o treino e infla artificialmente os resultados.

Se um atributo tiver variância zero (é constante), ele não traz informação e deve ser removido, ou então seu $\sigma_j$ deve ser tratado para evitar divisão por zero.

### 2.2 Atributos categóricos

Os algoritmos deste projeto trabalham com números. Atributos categóricos (como "cor = {vermelho, azul, verde}") precisam ser convertidos:

- **One-hot encoding:** cada categoria vira uma coluna binária (0/1). Esta forma é a mais adequada para atributos **nominais**, que não têm ordem.
- **Codificação ordinal:** cada categoria vira um inteiro (0, 1, 2…). Só faz sentido quando existe uma **ordem natural** entre as categorias, como "baixo < médio < alto".

Atenção: o one-hot aumenta o número de atributos $d$, o que afeta o R² ajustado e a matriz de covariância do Bayesiano multivariado. Colunas one-hot são linearmente dependentes entre si (a soma delas é sempre 1), o que pode tornar $\Sigma$ ou $\mathbf{X}^t\mathbf{X}$ singulares (ver Seções 6.5 e 7.4).

### 2.3 Valores ausentes

As alternativas mais simples são remover as amostras incompletas ou imputar um valor, como a média ou a mediana do atributo. Assim como na normalização, a imputação deve ser calculada **no treino** de cada fold.

---

## 3. k-Nearest Neighbors (kNN)

### 3.1 Ideia central

O kNN é um algoritmo **baseado em instâncias** (*lazy learning*, ou aprendizado preguiçoso). Ele **não constrói um modelo explícito**: o "treino" consiste apenas em armazenar os dados. Para prever a saída de uma nova amostra $\mathbf{x}$, o algoritmo segue três passos:

1. Calcula a distância entre $\mathbf{x}$ e **todas** as amostras de treino.
2. Seleciona as $k$ amostras mais próximas, que são os $k$ vizinhos.
3. Combina as saídas desses vizinhos para produzir a previsão.

A hipótese por trás do método é que **amostras parecidas (próximas no espaço de atributos) têm saídas parecidas**.

### 3.2 Medidas de distância

**Distância Euclidiana** (norma $L_2$). É a distância "em linha reta" entre dois pontos:

$$
d_{\text{Euclidiana}}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}
$$

- Por elevar as diferenças ao quadrado, **penaliza mais as diferenças grandes**. Um único atributo muito diferente pesa bastante no resultado.
- Gera regiões de vizinhança circulares (esféricas).

**Distância Manhattan** (norma $L_1$, ou *city block*). É a soma das diferenças absolutas, como andar por quarteirões de uma cidade:

$$
d_{\text{Manhattan}}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{d} |x_i - y_i|
$$

- Trata todas as diferenças de forma linear. Por isso é **mais robusta a outliers** e a atributos individuais com valores extremos.
- Em alta dimensionalidade, costuma discriminar melhor entre vizinhos próximos e distantes do que a Euclidiana.
- Gera regiões de vizinhança em forma de losango.

**Generalização (Minkowski):** as duas distâncias são casos particulares de

$$
d_p(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{d} |x_i - y_i|^p \right)^{1/p}
$$

com $p = 1$ para a Manhattan e $p = 2$ para a Euclidiana.

> **Observação:** para encontrar os vizinhos mais próximos, basta comparar a **distância Euclidiana ao quadrado**. A raiz quadrada é monotônica e não altera a ordenação, então pode ser omitida para economizar tempo.

### 3.3 kNN para classificação

A classe prevista é a **mais frequente** entre os $k$ vizinhos (voto majoritário):

$$
\hat{y} = \arg\max_{c} \sum_{i \in N_k(\mathbf{x})} \mathbb{1}(y_i = c)
$$

Aqui, $N_k(\mathbf{x})$ é o conjunto dos $k$ vizinhos e $\mathbb{1}(\cdot)$ é a função indicadora, que vale 1 se a condição é verdadeira e 0 caso contrário.

**Empates** podem acontecer de duas formas:

- **Empate de votos:** duas classes recebem o mesmo número de votos. Para evitar isso em problemas binários, usa-se $k$ ímpar. Em problemas multiclasse, as estratégias comuns são escolher a classe do vizinho mais próximo, reduzir $k$ até desempatar ou usar votação ponderada.
- **Empate de distâncias:** vários pontos estão à mesma distância. Basta adotar um critério determinístico, por exemplo a ordem original dos dados.

**Votação ponderada (opcional):** cada vizinho vota com peso $w_i = 1/d(\mathbf{x}, \mathbf{x}_i)$, dando mais importância aos vizinhos mais próximos.

### 3.4 kNN para regressão

A previsão é a **média dos valores-alvo** dos $k$ vizinhos:

$$
\hat{y} = \frac{1}{k} \sum_{i \in N_k(\mathbf{x})} y_i
$$

A versão ponderada pela distância é:

$$
\hat{y} = \frac{\sum_{i \in N_k} w_i \, y_i}{\sum_{i \in N_k} w_i}, \qquad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}
$$

### 3.5 Escolha do hiperparâmetro $k$

O valor de $k$ controla o equilíbrio entre **viés e variância**:

- **$k$ pequeno** (por exemplo, $k=1$): a fronteira de decisão fica muito irregular e o modelo é sensível a ruído. O resultado é **baixa variância no treino, alta variância no teste**, ou seja, overfitting.
- **$k$ grande:** a fronteira fica suave, mas o modelo pode ignorar padrões locais (underfitting). No limite $k = n$, ele sempre prevê a classe majoritária ou a média global.

Uma heurística comum é começar com $k \approx \sqrt{n}$. O ideal, porém, é escolher $k$ testando valores como 1, 3, 5, 7, … e comparando o desempenho. Essa escolha deve ser feita **apenas com dados de treino**, por exemplo com uma validação interna, para não contaminar a avaliação final.

### 3.6 Custo computacional

| Fase | Custo | Comentário |
|---|---|---|
| Treino | $O(1)$ (só armazena os dados) | extremamente rápido |
| Teste | $O(n_{\text{treino}} \cdot d)$ por amostra | lento: compara com todo o treino |
| Memória | $O(n \cdot d)$ | precisa guardar todo o conjunto de treino |

Por isso, no kNN o **tempo de teste é muito maior que o tempo de treino**, o oposto dos modelos paramétricos. A tabela de resultados deve deixar isso evidente.

### 3.7 Limitações

- **Sensível à escala** dos atributos, o que torna a normalização obrigatória.
- **Maldição da dimensionalidade:** em muitas dimensões, as distâncias entre os pontos tendem a ficar parecidas e o conceito de "vizinho próximo" perde o sentido.
- Sensível a atributos irrelevantes, que somam ruído à distância.
- Previsão lenta em datasets grandes.

---

## 4. Teoria de Decisão Bayesiana

Os dois classificadores bayesianos do projeto se baseiam no **Teorema de Bayes** e na **regra de decisão de mínimo erro**.

### 4.1 Teorema de Bayes

$$
P(\omega_k \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \omega_k) \, P(\omega_k)}{p(\mathbf{x})}
$$

| Termo | Nome | Significado |
|---|---|---|
| $P(\omega_k \mid \mathbf{x})$ | **Probabilidade a posteriori** | probabilidade de a amostra pertencer à classe $\omega_k$, dado que observamos $\mathbf{x}$ |
| $p(\mathbf{x} \mid \omega_k)$ | **Verossimilhança** (*likelihood*) | quão provável é observar $\mathbf{x}$ se a classe for $\omega_k$; é uma **densidade de probabilidade** |
| $P(\omega_k)$ | **Probabilidade a priori** | frequência da classe antes de observar $\mathbf{x}$ |
| $p(\mathbf{x})$ | **Evidência** | $\sum_{j} p(\mathbf{x} \mid \omega_j) P(\omega_j)$; é a mesma para todas as classes |

### 4.2 Regra de decisão (MAP — Máximo a Posteriori)

Para minimizar a probabilidade de erro, escolhemos a classe com **maior probabilidade a posteriori**:

$$
\hat{y} = \arg\max_{k} P(\omega_k \mid \mathbf{x}) = \arg\max_{k} \; p(\mathbf{x} \mid \omega_k) \, P(\omega_k)
$$

A evidência $p(\mathbf{x})$ pode ser ignorada porque é igual para todas as classes e não muda qual delas vence.

### 4.3 Estimação das probabilidades

Os dois termos da regra MAP são estimados a partir dos dados de treino:

- **Priori:** é a proporção de cada classe no treino,
  $$
  \hat{P}(\omega_k) = \frac{n_k}{n}
  $$
  em que $n_k$ é o número de amostras da classe $k$.
- **Verossimilhança:** supomos que $p(\mathbf{x} \mid \omega_k)$ segue uma **distribuição normal (Gaussiana)** e estimamos seus parâmetros (média e variância/covariância) com as amostras de cada classe. Esse é o chamado **método paramétrico**. A diferença entre os casos univariado e multivariado está justamente em **como essa normal é modelada**.

### 4.4 Funções discriminantes e uso do logaritmo

Densidades multiplicadas umas pelas outras geram números extremamente pequenos, que causam **underflow numérico** (o computador arredonda para 0). A solução é trabalhar com o **logaritmo**, que é uma função monotônica crescente e por isso não altera o $\arg\max$:

$$
g_k(\mathbf{x}) = \ln p(\mathbf{x} \mid \omega_k) + \ln P(\omega_k)
$$

$$
\hat{y} = \arg\max_k \; g_k(\mathbf{x})
$$

A função $g_k(\mathbf{x})$ é chamada de **função discriminante** da classe $k$. Com o log, produtos viram somas, o que também é mais eficiente.

---

## 5. Classificador Bayesiano — Caso Univariado

### 5.1 Ideia

Cada atributo é modelado **separadamente** por uma normal **univariada**, uma para cada par (atributo, classe). Para combinar os atributos, assumimos que eles são **condicionalmente independentes dada a classe**. Essa é a hipótese do **Naive Bayes Gaussiano**.

### 5.2 Densidade normal univariada

$$
p(x) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left[ -\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2 \right]
$$

- $\mu$ é a média, ou seja, o centro da curva.
- $\sigma$ é o desvio padrão, que define a largura da curva; $\sigma^2$ é a variância.
- O termo $\left(\frac{x-\mu}{\sigma}\right)^2$ mede quantos desvios padrão $x$ está afastado da média.

### 5.3 Hipótese de independência condicional

$$
p(\mathbf{x} \mid \omega_k) = \prod_{j=1}^{d} p(x_j \mid \omega_k)
$$

Em palavras: **dentro de cada classe**, conhecer o valor de um atributo não traz informação sobre os outros. Isso equivale a supor que a matriz de covariância de cada classe é **diagonal**, ou seja, que não há correlação entre os atributos.

Essa hipótese raramente é verdadeira na prática, daí o nome "ingênuo" (*naive*). Mesmo assim, o classificador costuma funcionar bem, porque para acertar a classe basta que o **ranking** das posterioris esteja correto, não seus valores exatos.

### 5.4 Treinamento (estimação de parâmetros)

Para cada classe $\omega_k$ e cada atributo $j$, os parâmetros são estimados por **máxima verossimilhança** usando apenas as $n_k$ amostras da classe:

$$
\hat{\mu}_{kj} = \frac{1}{n_k} \sum_{i:\, y_i = \omega_k} x_{ij}
$$

$$
\hat{\sigma}^2_{kj} = \frac{1}{n_k} \sum_{i:\, y_i = \omega_k} \left( x_{ij} - \hat{\mu}_{kj} \right)^2
$$

O estimador de máxima verossimilhança divide por $n_k$. A versão não viesada divide por $n_k - 1$. Com mais de 1000 amostras, a diferença é desprezível.

No total, são estimados $C \times d$ médias, $C \times d$ variâncias e $C$ prioris. São **poucos parâmetros**, o que torna o modelo rápido e pouco propenso a overfitting.

### 5.5 Classificação

Aplicando o log e a independência, a função discriminante vira uma **soma**:

$$
g_k(\mathbf{x}) = \ln P(\omega_k) + \sum_{j=1}^{d} \left[ -\frac{1}{2}\ln(2\pi\sigma^2_{kj}) - \frac{(x_j - \mu_{kj})^2}{2\sigma^2_{kj}} \right]
$$

$$
\hat{y} = \arg\max_k \; g_k(\mathbf{x})
$$

### 5.6 Cuidados

- **Variância zero:** se um atributo for constante dentro de uma classe, $\sigma^2_{kj} = 0$ e a densidade se torna indefinida (divisão por zero). A solução usual é somar um valor pequeno $\varepsilon$ a todas as variâncias (*variance smoothing*), por exemplo $\varepsilon = 10^{-9}$ multiplicado pela maior variância observada.
- **Normalização:** não é necessária para o Bayesiano, porque cada atributo tem sua própria $\mu$ e $\sigma$. Ela também não prejudica o modelo.
- **Atributos não-gaussianos:** variáveis muito assimétricas ou discretas violam a hipótese de normalidade e podem piorar o desempenho.

### 5.7 Custo computacional

- **Treino:** $O(n \cdot d)$, uma única passada nos dados. É muito rápido.
- **Teste:** $O(C \cdot d)$ por amostra. Também é muito rápido.

---

## 6. Classificador Bayesiano — Caso Multivariado

### 6.1 Ideia

Em vez de tratar cada atributo isoladamente, modelamos o vetor $\mathbf{x}$ inteiro, de cada classe, com uma **normal multivariada**. Assim o modelo captura as **correlações entre os atributos**, que o caso univariado ignora.

### 6.2 Densidade normal multivariada

$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} \, |\boldsymbol{\Sigma}|^{1/2}} \exp\left[ -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^t \, \boldsymbol{\Sigma}^{-1} \, (\mathbf{x} - \boldsymbol{\mu}) \right]
$$

| Elemento | Dimensão | Significado |
|---|---|---|
| $\mathbf{x}$ | $d \times 1$ | vetor de atributos da amostra |
| $\boldsymbol{\mu}$ | $d \times 1$ | vetor de médias (centro da distribuição) |
| $\boldsymbol{\Sigma}$ | $d \times d$ | matriz de covariância (forma e orientação da distribuição) |
| $\lvert\boldsymbol{\Sigma}\rvert$ | escalar | determinante: mede o "volume" ocupado pela distribuição |
| $\boldsymbol{\Sigma}^{-1}$ | $d \times d$ | inversa da covariância (também chamada matriz de precisão) |
| $(\mathbf{x}-\boldsymbol{\mu})^t$ | $1 \times d$ | transposta do vetor de desvios |

### 6.3 A matriz de covariância

$$
\boldsymbol{\Sigma} =
\begin{bmatrix}
\sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1d} \\
\sigma_{21} & \sigma_2^2 & \cdots & \sigma_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{d1} & \sigma_{d2} & \cdots & \sigma_d^2
\end{bmatrix}
$$

- **Diagonal principal** ($\sigma_j^2$): variância de cada atributo.
- **Fora da diagonal** ($\sigma_{ij}$): covariância entre os atributos $i$ e $j$. Um valor positivo indica que os atributos crescem juntos; um negativo, que um cresce enquanto o outro diminui; zero indica ausência de correlação linear.
- É **simétrica** ($\sigma_{ij} = \sigma_{ji}$) e **semidefinida positiva**.

A covariância amostral entre os atributos $i$ e $j$, usando as amostras de uma classe, é:

$$
\hat{\sigma}_{ij} = \frac{1}{n_k - 1} \sum_{m=1}^{n_k} (x_{mi} - \hat{\mu}_i)(x_{mj} - \hat{\mu}_j)
$$

Na forma matricial, com $\mathbf{X}_k$ sendo as amostras da classe $k$:

$$
\hat{\boldsymbol{\Sigma}}_k = \frac{1}{n_k - 1} \sum_{m=1}^{n_k} (\mathbf{x}_m - \hat{\boldsymbol{\mu}}_k)(\mathbf{x}_m - \hat{\boldsymbol{\mu}}_k)^t
$$

Esse é o cálculo realizado pelo `np.cov` citado no enunciado. Por padrão, ele divide por $n-1$ e espera as **variáveis nas linhas**; com os dados organizados com amostras nas linhas, é preciso transpor a matriz ou indicar que as variáveis estão nas colunas.

> **Relação com o caso univariado:** se $\boldsymbol{\Sigma}$ for **diagonal** (todas as covariâncias iguais a zero), a normal multivariada se fatora no produto das normais univariadas. Ou seja, **o caso univariado (Naive Bayes) é um caso particular do multivariado**.

### 6.4 Treinamento e classificação

**Treino:** para cada classe $k$, estimamos:

- a priori $\hat{P}(\omega_k) = n_k / n$;
- o vetor de médias $\hat{\boldsymbol{\mu}}_k$;
- a matriz de covariância $\hat{\boldsymbol{\Sigma}}_k$, e com ela pré-calculamos $\hat{\boldsymbol{\Sigma}}_k^{-1}$ e $\ln|\hat{\boldsymbol{\Sigma}}_k|$.

**Classificação:** aplicando o logaritmo à densidade, obtemos a função discriminante:

$$
g_k(\mathbf{x}) = -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_k)^t \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) - \frac{1}{2} \ln |\boldsymbol{\Sigma}_k| + \ln P(\omega_k) \;\; \underbrace{- \frac{d}{2}\ln(2\pi)}_{\text{constante, pode ser omitida}}
$$

$$
\hat{y} = \arg\max_k \; g_k(\mathbf{x})
$$

**Distância de Mahalanobis.** O primeiro termo contém a **distância de Mahalanobis ao quadrado**:

$$
D_M^2(\mathbf{x}, \boldsymbol{\mu}_k) = (\mathbf{x} - \boldsymbol{\mu}_k)^t \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)
$$

Ela é uma distância "inteligente" que leva em conta a **escala** e a **correlação** dos atributos. Se $\boldsymbol{\Sigma} = \mathbf{I}$ (matriz identidade), ela se reduz à distância Euclidiana. Intuitivamente, o classificador escolhe a classe cujo centro está mais próximo de $\mathbf{x}$ segundo Mahalanobis, com correções pelo volume da distribuição ($\ln|\boldsymbol{\Sigma}_k|$) e pela frequência da classe ($\ln P(\omega_k)$).

**Fronteiras de decisão:** como cada classe tem sua própria $\boldsymbol{\Sigma}_k$, as fronteiras entre as classes são **quádricas**: elipses, parábolas ou hipérboles. Por isso este modelo também é conhecido como **Análise Discriminante Quadrática (QDA)**. Se todas as classes compartilhassem a mesma $\boldsymbol{\Sigma}$, os termos quadráticos se cancelariam e as fronteiras seriam lineares; esse caso é a **LDA**.

### 6.5 Problemas numéricos e regularização

O cálculo exige **inverter** $\boldsymbol{\Sigma}_k$ e calcular seu **determinante**, o que pode falhar em quatro situações:

1. **Matriz singular** ($|\boldsymbol{\Sigma}| = 0$, sem inversa). Isso acontece quando:
   - há atributos **constantes** dentro de uma classe;
   - há atributos **linearmente dependentes**, como colunas duplicadas, colunas one-hot completas ou um atributo que é soma de outros;
   - a classe tem **poucas amostras**: se $n_k \le d$, a matriz é necessariamente singular.
2. **Matriz mal-condicionada:** o determinante é quase zero e a inversa fica numericamente instável.
3. **Underflow/overflow no determinante:** em dimensão alta, $|\boldsymbol{\Sigma}|$ pode ser um número minúsculo ou gigantesco. O ideal é calcular diretamente o **log-determinante** $\ln|\boldsymbol{\Sigma}|$, via fatoração (Cholesky ou LU), em vez de calcular o determinante e depois o log.
4. **Exponencial da densidade:** por isso trabalhamos com $g_k(\mathbf{x})$ (o log), nunca com $p(\mathbf{x})$ diretamente.

**Regularização (shrinkage):** a solução padrão para a singularidade é somar um pequeno valor à diagonal da matriz:

$$
\boldsymbol{\Sigma}_k^{\text{reg}} = \boldsymbol{\Sigma}_k + \lambda \mathbf{I}, \qquad \lambda \text{ pequeno (ex.: } 10^{-6}\text{ a } 10^{-3}\text{)}
$$

Isso garante que a matriz seja invertível, à custa de um viés pequeno. Outras alternativas são remover atributos redundantes ou usar a pseudo-inversa.

### 6.6 Custo e quantidade de parâmetros

- **Parâmetros:** para cada classe, são $d$ médias mais $\frac{d(d+1)}{2}$ covariâncias. Com $d = 20$ e $C = 3$, isso dá $3 \times (20 + 210) = 690$ parâmetros, contra $120$ no caso univariado. Mais parâmetros exigem **mais dados** para uma estimativa confiável e trazem **maior risco de overfitting**.
- **Treino:** $O(n \cdot d^2 + C \cdot d^3)$. O termo $d^3$ vem da inversão de cada matriz.
- **Teste:** $O(C \cdot d^2)$ por amostra. É mais lento que o univariado, mas muito mais rápido que o kNN.

### 6.7 Univariado × Multivariado

| Aspecto | Univariado (Naive) | Multivariado (QDA) |
|---|---|---|
| Correlação entre atributos | ignora | modela |
| Covariância | diagonal | completa |
| Nº de parâmetros por classe | $2d$ | $d + d(d+1)/2$ |
| Dados necessários | poucos | muitos ($n_k \gg d$) |
| Fronteira de decisão | quadrática, porém alinhada aos eixos | quadrática arbitrária |
| Estabilidade numérica | alta | pode exigir regularização |
| Velocidade | muito rápido | rápido |

---

## 7. Regressão Linear Múltipla

### 7.1 Modelo

A Regressão Linear Múltipla assume que o alvo é uma **combinação linear** dos atributos, somada a um erro aleatório:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_d x_d + \varepsilon
$$

- $\beta_0$ é o **intercepto (viés)**: o valor de $y$ quando todos os atributos valem zero.
- $\beta_j$ é o **coeficiente** do atributo $j$: a variação esperada em $y$ quando $x_j$ aumenta uma unidade e **os demais atributos ficam constantes**.
- $\varepsilon$ é o erro, ou ruído, que o modelo não explica.

A previsão do modelo é $\hat{y} = \beta_0 + \sum_j \beta_j x_j$.

### 7.2 Forma matricial

Adicionando uma **coluna de 1s** à matriz de dados para representar o intercepto:

$$
\mathbf{X} =
\begin{bmatrix}
1 & x_{11} & \cdots & x_{1d} \\
1 & x_{21} & \cdots & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & \cdots & x_{nd}
\end{bmatrix}_{n \times (d+1)},
\qquad
\boldsymbol{\beta} =
\begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_d \end{bmatrix},
\qquad
\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}
$$

### 7.3 Estimação por Mínimos Quadrados Ordinários (MQO / OLS)

Buscamos os coeficientes que **minimizam a soma dos quadrados dos resíduos** (SSE):

$$
\text{SSE}(\boldsymbol{\beta}) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^t (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$

Derivando em relação a $\boldsymbol{\beta}$ e igualando a zero, obtemos as **equações normais**:

$$
\mathbf{X}^t \mathbf{X} \, \boldsymbol{\beta} = \mathbf{X}^t \mathbf{y}
$$

E daí, a solução em forma fechada:

$$
\boxed{\hat{\boldsymbol{\beta}} = (\mathbf{X}^t \mathbf{X})^{-1} \mathbf{X}^t \mathbf{y}}
$$

Como a função SSE é convexa, essa é a **solução ótima global**. Não é preciso nenhum algoritmo iterativo.

**Interpretação geométrica:** $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$ é a **projeção ortogonal** de $\mathbf{y}$ no espaço gerado pelas colunas de $\mathbf{X}$. Os resíduos resultantes são ortogonais a todas as colunas de $\mathbf{X}$.

### 7.4 Cuidados numéricos

- **Multicolinearidade:** atributos fortemente correlacionados tornam $\mathbf{X}^t\mathbf{X}$ quase singular. Os coeficientes ficam instáveis e com sinais estranhos, embora as previsões possam continuar razoáveis.
- **Singularidade:** com atributos linearmente dependentes (por exemplo, todas as colunas one-hot junto com o intercepto), $\mathbf{X}^t\mathbf{X}$ não tem inversa. As soluções possíveis são:
  - remover uma coluna de cada grupo one-hot (*drop-first*);
  - usar a **pseudo-inversa de Moore-Penrose**, $\hat{\boldsymbol{\beta}} = \mathbf{X}^{+}\mathbf{y}$;
  - resolver o sistema linear das equações normais por fatoração (QR, Cholesky), em vez de inverter a matriz explicitamente, o que também é numericamente mais estável;
  - usar regularização **Ridge**: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^t\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^t\mathbf{y}$. É a mesma ideia do $\lambda\mathbf{I}$ usado no Bayesiano multivariado. O intercepto normalmente não é penalizado.
- **Normalização:** não altera as previsões do OLS, mas melhora o condicionamento numérico e torna os coeficientes comparáveis entre si.

### 7.5 Pressupostos clássicos

Os pressupostos abaixo garantem as propriedades estatísticas do estimador. Na prática, servem para **interpretar** os resultados e as limitações do modelo:

1. **Linearidade:** a relação entre os atributos e o alvo é linear.
2. **Independência** dos erros.
3. **Homocedasticidade:** a variância do erro é constante.
4. **Normalidade** dos erros. Esse pressuposto importa para inferência estatística, não para a previsão.
5. **Ausência de multicolinearidade perfeita.**

Se a relação real for não-linear, a regressão linear terá **viés alto** (underfitting). Nesse cenário, o kNN, que não assume nenhuma forma funcional, pode se sair melhor.

### 7.6 Custo computacional

- **Treino:** $O(n \cdot d^2 + d^3)$, para montar $\mathbf{X}^t\mathbf{X}$ e resolver o sistema.
- **Teste:** $O(d)$ por amostra, apenas um produto escalar. É **extremamente rápido**.

---

## 8. Validação Cruzada k-Fold

### 8.1 Motivação

Uma única divisão treino/teste dá uma estimativa de desempenho que **depende muito da sorte** de quais amostras caíram em cada lado. A validação cruzada usa **todos os dados** tanto para treinar quanto para testar, em rodadas diferentes. O resultado é uma estimativa mais confiável e acompanhada de uma medida de **variabilidade**, o desvio padrão.

### 8.2 Procedimento (k-fold, com k = 5 por exemplo)

1. **Embaralhar** os dados aleatoriamente, com uma semente fixa para que o experimento seja reprodutível.
2. Dividir os dados em $K$ partes (*folds*) de tamanho aproximadamente igual.
3. Para cada $i = 1, \dots, K$:
   - o fold $i$ é o **conjunto de teste**;
   - os outros $K - 1$ folds formam o **conjunto de treino**;
   - ajustar o pré-processamento **somente no treino** (normalização, imputação, etc.);
   - **treinar** o modelo, medindo o tempo de treino;
   - **prever** o conjunto de teste, medindo o tempo de teste;
   - calcular as métricas.
4. Ao final, há $K$ valores de cada métrica. Calcula-se a **média** e o **desvio padrão** desses valores.

```
Fold 1: [TESTE] [treino] [treino] [treino] [treino]
Fold 2: [treino] [TESTE] [treino] [treino] [treino]
Fold 3: [treino] [treino] [TESTE] [treino] [treino]
Fold 4: [treino] [treino] [treino] [TESTE] [treino]
Fold 5: [treino] [treino] [treino] [treino] [TESTE]
```

Cada amostra é usada **exatamente uma vez** como teste e $K-1$ vezes como treino.

### 8.3 Validação cruzada estratificada (classificação)

Na classificação, especialmente com **classes desbalanceadas**, convém que cada fold mantenha a **mesma proporção de classes** do dataset completo. Para isso, as amostras são divididas por classe e cada classe é distribuída igualmente entre os folds. Isso evita folds sem nenhuma amostra de uma classe rara, que gerariam métricas instáveis.

### 8.4 Estatísticas finais

Dados os valores $m_1, \dots, m_K$ de uma métrica, um para cada fold:

$$
\bar{m} = \frac{1}{K} \sum_{i=1}^{K} m_i
\qquad\qquad
s = \sqrt{\frac{1}{K - 1} \sum_{i=1}^{K} (m_i - \bar{m})^2}
$$

O resultado é reportado como $\bar{m} \pm s$, por exemplo $0.85 \pm 0.03$. Use o desvio padrão **amostral** (dividindo por $K-1$) ou o **populacional** (dividindo por $K$), mas deixe claro qual foi usado.

- **Média:** é o desempenho esperado do modelo.
- **Desvio padrão:** mede a **estabilidade** do modelo. Um desvio alto indica que o desempenho depende muito de quais dados foram usados no treino.

### 8.5 Medição dos tempos

- **Tempo de treino:** duração da etapa de ajuste do modelo. No kNN, é apenas o armazenamento dos dados; nos Bayesianos, a estimação de $\mu$, $\sigma$ e $\Sigma$; na regressão, a resolução das equações normais.
- **Tempo de teste:** duração da etapa de previsão de todo o fold de teste.
- Use um relógio de alta resolução e meça **apenas** a etapa correspondente, sem incluir a leitura dos dados ou o cálculo de métricas.
- O tempo também é reportado como média ± desvio padrão entre os folds.

### 8.6 Vazamento de dados (data leakage)

Qualquer informação do conjunto de teste que influencie o treino **invalida a avaliação**. Os erros mais comuns são:

- normalizar ou imputar usando o dataset inteiro antes de dividir os folds;
- escolher $k$, $\lambda$ ou atributos olhando o desempenho no fold de teste;
- ter amostras duplicadas entre treino e teste.

---

## 9. Métricas de Classificação

### 9.1 Matriz de confusão

Todas as métricas de classificação derivam da **matriz de confusão**. No caso binário, tomando uma classe como "positiva":

|  | **Previsto: Positivo** | **Previsto: Negativo** |
|---|---|---|
| **Real: Positivo** | VP (Verdadeiro Positivo) | FN (Falso Negativo) |
| **Real: Negativo** | FP (Falso Positivo) | VN (Verdadeiro Negativo) |

No caso multiclasse, a matriz é $C \times C$: o elemento $(i, j)$ conta quantas amostras da classe real $i$ foram previstas como classe $j$. A diagonal contém os acertos.

### 9.2 Acurácia (Accuracy)

É a proporção de previsões corretas:

$$
\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN} = \frac{\text{nº de acertos}}{n}
$$

- É intuitiva, mas **enganosa com classes desbalanceadas**. Num dataset com 95% de uma classe, um modelo que sempre prevê essa classe atinge 95% de acurácia sem aprender nada.

### 9.3 Precisão (Precision)

Dos exemplos que o modelo **previu como positivos**, quantos realmente são positivos?

$$
\text{Precisão} = \frac{VP}{VP + FP}
$$

É importante quando o **falso positivo é caro**. Por exemplo, num filtro de spam, marcar um e-mail legítimo como spam é grave.

### 9.4 Revocação (Recall / Sensibilidade)

Dos exemplos que **realmente são positivos**, quantos o modelo encontrou?

$$
\text{Recall} = \frac{VP}{VP + FN}
$$

É importante quando o **falso negativo é caro**. Por exemplo, num diagnóstico médico, não detectar uma doença é grave.

### 9.5 F1-Score

É a **média harmônica** entre precisão e recall:

$$
F_1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}} = \frac{2\,VP}{2\,VP + FP + FN}
$$

A média harmônica **penaliza desequilíbrios**: se uma das duas métricas for baixa, o F1 também será baixo. Isso faz do F1 uma métrica mais informativa que a acurácia quando as classes são desbalanceadas.

### 9.6 Extensão para multiclasse

Precisão, recall e F1 são calculados **por classe**, adotando a estratégia "uma classe contra o resto", e depois **agregados**:

- **Macro-average:** média simples das métricas de cada classe. Todas as classes pesam igual, inclusive as raras.
  $$
  \text{Precisão}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \text{Precisão}_c
  $$
- **Weighted-average:** média ponderada pelo número de amostras reais de cada classe (o *suporte*).
- **Micro-average:** soma VP, FP e FN de todas as classes e só então calcula a métrica. Em problemas multiclasse com um único rótulo por amostra, a micro-precisão, o micro-recall e o micro-F1 são **iguais à acurácia**.

> **Recomendação:** use **macro-average** e indique isso nos slides. Ela evidencia se o modelo está ignorando alguma classe minoritária.

**Caso de borda:** se uma classe nunca é prevista, temos $VP + FP = 0$ e a precisão fica indefinida. A convenção usual é atribuir 0 nesse caso.

---

## 10. Métricas de Regressão

### 10.1 Erro Absoluto Médio (MAE)

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

- Está na **mesma unidade** do alvo. Por exemplo, "o modelo erra em média R\$ 12.000".
- Trata todos os erros linearmente, então é **robusto a outliers**.
- Quanto **menor**, melhor.

### 10.2 Erro Quadrático Médio (MSE) e RMSE (complementares)

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}
$$

Essas métricas penalizam fortemente os erros grandes. O enunciado não as exige, mas o MSE aparece no cálculo do R².

### 10.3 Coeficiente de Determinação (R²)

O R² mede a **proporção da variância do alvo que o modelo explica**:

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}
$$

- $SS_{\text{res}}$ é a soma dos quadrados dos **resíduos**, ou seja, o erro do modelo.
- $SS_{\text{tot}}$ é a soma total dos quadrados: o erro de um modelo "ingênuo" que sempre prevê a média $\bar{y}$.

**Interpretação:**

| Valor | Significado |
|---|---|
| $R^2 = 1$ | previsão perfeita |
| $R^2 = 0$ | o modelo é tão bom quanto prever sempre a média |
| $R^2 < 0$ | o modelo é **pior** que prever a média (pode acontecer no teste) |

Na validação cruzada, o $\bar{y}$ usado é a média do **fold de teste**.

### 10.4 R² Ajustado

O R² comum **nunca diminui** quando adicionamos atributos, mesmo que eles sejam ruído puro. O R² ajustado **penaliza a complexidade**, isto é, o número de preditores:

$$
R^2_{\text{ajustado}} = 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1}
$$

- $n$ é o número de amostras **do conjunto avaliado** (o fold de teste).
- $p$ é o número de **atributos preditores**, sem contar o intercepto e já considerando as colunas criadas pelo one-hot.

**Propriedades:**

- $R^2_{\text{ajustado}} \le R^2$.
- Só aumenta com um novo atributo se esse atributo melhorar o modelo mais do que o esperado por acaso.
- Pode ser negativo.
- Com $n \gg p$, por exemplo 1000 amostras e 15 atributos, os dois valores ficam muito próximos.

> **Observação sobre o kNN:** o R² ajustado foi concebido para modelos lineares, em que $p$ é o número de parâmetros estimados. Para o kNN, que não tem parâmetros nesse sentido, a convenção prática é usar $p$ = número de atributos, para manter a comparação uniforme. Vale mencionar isso nos slides.

---

## 11. Comparação entre os algoritmos

### 11.1 Resumo das características

| Algoritmo | Tipo | Hipótese principal | Treino | Teste | Pontos fortes | Limitações |
|---|---|---|---|---|---|---|
| kNN Euclidiano | não paramétrico, *lazy* | vizinhos próximos têm saídas semelhantes | muito rápido | **lento** | simples, sem suposição sobre a forma dos dados, fronteiras flexíveis | sensível à escala, a outliers e à alta dimensionalidade; previsão cara |
| kNN Manhattan | não paramétrico, *lazy* | idem | muito rápido | **lento** | mais robusto a outliers e a alta dimensão que o Euclidiano | mesmas limitações gerais do kNN |
| Bayes Univariado | paramétrico, generativo | atributos gaussianos e **independentes** dada a classe | muito rápido | muito rápido | poucos parâmetros, estável, funciona com poucos dados | ignora correlações; sensível à não-normalidade |
| Bayes Multivariado | paramétrico, generativo | atributos com distribuição **normal multivariada** por classe | rápido | rápido | captura correlações; fronteiras quadráticas | muitos parâmetros; $\Sigma$ pode ser singular; exige $n_k \gg d$ |
| Regressão Linear Múltipla | paramétrico, discriminativo | relação **linear** entre atributos e alvo | rápido | **muito rápido** | interpretável, solução exata, previsão instantânea | não captura relações não-lineares; sensível à multicolinearidade e a outliers |

### 11.2 Questões para guiar a análise crítica

- **Desempenho:** qual modelo obteve as maiores Acurácia e F1 (classificação) e o maior R² e menor MAE (regressão)? As diferenças entre os modelos são maiores que os desvios padrão? Se as faixas $\bar{m} \pm s$ se sobrepõem bastante, a diferença pode não ser significativa.
- **Euclidiana × Manhattan:** houve diferença? Se houve, isso pode indicar a presença de outliers ou de muitas dimensões.
- **Univariado × Multivariado:** se o multivariado foi melhor, os atributos são correlacionados e essa correlação é informativa. Se foi pior, provavelmente faltaram dados para estimar $\Sigma$ ou a matriz ficou mal-condicionada.
- **kNN × Regressão Linear:** se o kNN foi melhor, a relação entre os atributos e o alvo provavelmente é não-linear. Se a regressão linear foi melhor ou equivalente, a relação é aproximadamente linear, e a regressão vence em velocidade e interpretabilidade.
- **Eficiência:** o kNN tem treino quase instantâneo e teste caro. Os modelos paramétricos invertem essa relação. Em produção, o tempo de **teste** costuma importar mais, pois o modelo é treinado uma vez e usado muitas vezes.
- **Equilíbrio:** o modelo mais preciso compensa seu custo computacional? Uma pequena perda de desempenho pode valer a pena se o ganho de velocidade for grande.

---

## 12. Referências

- DUDA, R. O.; HART, P. E.; STORK, D. G. *Pattern Classification*. 2. ed. Wiley, 2001. — Teoria de decisão bayesiana, normal multivariada, funções discriminantes, kNN.
- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Modelos generativos, distribuição gaussiana, regressão linear.
- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning*. 2. ed. Springer, 2009. — Validação cruzada, LDA/QDA, kNN, regressão linear, viés-variância.
- JAMES, G.; WITTEN, D.; HASTIE, T.; TIBSHIRANI, R. *An Introduction to Statistical Learning*. 2. ed. Springer, 2021. — Abordagem introdutória de todos os tópicos.
- MITCHELL, T. M. *Machine Learning*. McGraw-Hill, 1997. — Aprendizado baseado em instâncias e Naive Bayes.
- MONTGOMERY, D. C.; PECK, E. A.; VINING, G. G. *Introduction to Linear Regression Analysis*. 5. ed. Wiley, 2012. — Mínimos quadrados, R² e R² ajustado, multicolinearidade.
- OpenML. Disponível em: https://www.openml.org/ — Repositório dos datasets.
