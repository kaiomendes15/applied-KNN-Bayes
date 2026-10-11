# Guia de Estudo — Classificadores Bayesianos (Membro 3)

Este guia explica **tudo o que o Membro 3 implementou**, da ideia ao código, sem exigir conhecimento prévio de numpy. O objetivo é que você consiga **explicar qualquer linha** de `src/bayes.py` na apresentação.

**Onde está cada coisa:**

| O quê | Onde |
|---|---|
| Bayesiano univariado (`GaussianNaiveBayes`) | `src/bayes.py`, linhas 15–90 |
| Bayesiano multivariado (`GaussianBayes`) | `src/bayes.py`, linhas 93–193 |
| Teoria completa com fórmulas | `docs/fundamentacao_teorica.md`, seções 4 a 6 |
| Primeiro contato com o Naive Bayes | `docs/naive_bayes.md` |

---

## 1. O caminho completo: do arquivo à tabela

Antes dos detalhes, veja onde os modelos do Membro 3 entram no projeto:

```
credit-g.arff
   │  leitor ARFF: texto → números (one-hot nas colunas de categoria)
   ▼
X (1000 × 63) e y (good=0, bad=1)
   │  k-fold estratificado: divide em 5 partes, cada uma vira teste uma vez
   ▼
para cada uma das 5 rodadas:
   │  z-score: deixa cada coluna com média 0 e desvio 1 (calculado só no treino)
   │  modelo.fit(X_treino, y_treino)   ← Membro 3: aprende o "resumo" de cada classe
   │  modelo.predict(X_teste)          ← Membro 3: compara cada cliente com os resumos
   │  métricas: acurácia, precisão, recall, F1, tempos
   ▼
média ± desvio padrão das 5 rodadas → uma linha da tabela
```

O Membro 3 entrega só as duas caixas `fit` e `predict`. Todo o resto (leitura, divisão em folds, normalização, métricas) vem dos outros membros.

---

## 2. A base: Teorema de Bayes e regra MAP

O problema: dado um cliente com atributos **x** (duração, valor, idade…), ele é `good` ou `bad`?

O Teorema de Bayes diz:

```
P(classe | x)  =  P(classe) × p(x | classe)  /  p(x)
  posteriori        priori     verossimilhança    evidência
```

| Termo | Pergunta que responde |
|---|---|
| **Priori** P(classe) | Antes de olhar o cliente, quão comum é essa classe? (70% `good`, 30% `bad`) |
| **Verossimilhança** p(x \| classe) | Se o cliente fosse dessa classe, quão "típico" ele seria? |
| **Posteriori** P(classe \| x) | Depois de olhar o cliente, qual a chance dele ser dessa classe? |
| **Evidência** p(x) | É igual para todas as classes, então **pode ser ignorada** na comparação. |

**Regra MAP (Máximo A Posteriori):** calcula a pontuação `priori × verossimilhança` de cada classe e **escolhe a maior**.

### O que muda entre os dois classificadores

Os dois usam a mesma regra MAP e a mesma priori. A **única diferença** é como calculam a verossimilhança:

- **Univariado:** olha cada atributo **separadamente** (um sino por atributo).
- **Multivariado:** olha **todos os atributos juntos** (um "sino em várias dimensões", que enxerga a relação entre eles).

### O truque do logaritmo (usado nos dois)

Multiplicar dezenas de probabilidades pequenas gera números tão perto de zero que o computador os arredonda para 0 (*underflow*). Por isso o código trabalha com o **log da pontuação**:

- log transforma **multiplicação em soma**: log(a × b) = log(a) + log(b);
- log **preserva a ordem**: se A > B, então log(A) > log(B), então a classe vencedora é a mesma.

```
log-pontuação(classe) = log(priori) + log(verossimilhança)
```

---

## 3. O mínimo de numpy que você precisa

Pense num array do numpy como uma **planilha**. O **formato** (`shape`) diz o tamanho de cada dimensão.

| Formato | Como imaginar | Exemplo no projeto |
|---|---|---|
| `(n,)` | uma lista de n valores | `y`: a classe de cada cliente |
| `(n, d)` | uma tabela: n linhas (clientes) × d colunas (atributos) | `X` |
| `(C, d, d)` | uma pilha de C tabelas d × d | as matrizes Σ, uma por classe |

Notação usada no guia: **n** = nº de amostras, **d** = nº de atributos, **C** = nº de classes (2 no credit-g).

| Código | O que faz, em português |
|---|---|
| `np.asarray(X, dtype=float)` | Garante que X é um array de números decimais. |
| `np.unique(y)` | Lista os valores distintos de y, em ordem: `[0, 1]`. |
| `y == c` | Cria uma lista de Verdadeiro/Falso: "este cliente é da classe c?". |
| `X[y == c]` | Mantém **só as linhas** onde a resposta foi Verdadeiro (o "filtro" da planilha). |
| `len(Xk)` | Número de linhas. |
| `Xk.mean(axis=0)` | Média **de cada coluna** (`axis=0` = "descendo pelas linhas"). |
| `Xk.var(axis=0)` | Variância de cada coluna. |
| `np.empty((C, d))` | Reserva uma tabela vazia desse tamanho, que depois é preenchida. |
| `np.log`, `np.pi` | Logaritmo natural e o número π. |
| `a @ b` | Multiplicação de matrizes. |
| `soma(..., axis=1)` | Soma **cada linha** (`axis=1` = "atravessando as colunas"). |
| `np.argmax(scores, axis=1)` | Em cada linha, diz **em qual coluna** está o maior valor. |
| `np.cov(Xk, rowvar=False)` | Matriz de covariância (seção 5.2). |
| `np.eye(d)` | Matriz identidade: 1 na diagonal, 0 no resto. |
| `np.linalg.inv(S)` | Matriz inversa. |
| `np.linalg.slogdet(S)` | Logaritmo do determinante (seção 5.6). |

**Broadcasting** (usado no univariado): quando você faz `X - media`, o numpy "estica" automaticamente a média para subtrair de todas as linhas de uma vez, sem precisar de laço. É só isso: **operações em lote**.

---

## 4. Bayesiano Univariado (`GaussianNaiveBayes`)

### 4.1 A ideia

Para cada **classe** e cada **atributo**, o modelo guarda um sino (distribuição normal), descrito por dois números:

- **média (μ):** o valor típico daquele atributo naquela classe;
- **variância (σ²):** o quanto os valores se espalham em volta da média.

Ele supõe que, dentro de uma classe, os atributos **não se influenciam** (por isso "naive", ingênuo). Com essa suposição, a verossimilhança vira um produto de sinos, ou seja, **uma soma de logs**.

### 4.2 `fit`: aprendendo o resumo de cada classe

Em português, o `fit` faz:

```
classes = valores distintos de y                     → [good, bad]
para cada classe c:
    Xk = só os clientes da classe c
    priori[c]    = nº de clientes da classe / nº total
    média[c]     = média de cada coluna de Xk
    variância[c] = variância de cada coluna de Xk
no fim: soma um ε pequeno a todas as variâncias (var_smoothing)
```

O que fica guardado no credit-g (C = 2 classes, d = 63 atributos):

| Nome | Formato | Conteúdo |
|---|---|---|
| `classes` | (2,) | `[0, 1]` |
| `prior` | (2,) | `[0,7  0,3]` |
| `mean` | (2, 63) | linha 0 = médias dos `good`; linha 1 = médias dos `bad` |
| `var` | (2, 63) | mesmo formato, com as variâncias |

**Por que `k` e `c` no laço?** `c` é o **rótulo** da classe (usado no filtro `y == c`) e `k` é a **posição** na tabela (`mean[k]`). Eles são separados porque os rótulos nem sempre são 0, 1, 2…

### 4.3 `var_smoothing`: por que somar um ε

A fórmula do sino **divide pela variância**. Se uma coluna é constante dentro de uma classe (variância 0), a conta quebra.

Isso acontece muito com colunas **one-hot** (0 ou 1): por exemplo, uma categoria que nenhum cliente `bad` do treino tem gera uma coluna toda de zeros nessa classe.

A solução é somar um **ε** a todas as variâncias:

```
ε = var_smoothing × (maior variância entre as colunas de X)
```

Ele é proporcional à maior variância para se adaptar à escala dos dados.

**Por que usamos `var_smoothing = 0,01` e não o usual `1e-9`?**
Com ε minúsculo, a variância "quase zero" vira um sino finíssimo. Se um cliente de teste tiver essa categoria (valor 1 em vez de 0), a penalidade é gigante:

| ε | Penalidade para um desvio de 1 (≈ desvio² / variância ÷ 2) |
|---|---|
| 1e-9 | ≈ 500.000.000 |
| 0,01 | ≈ 50 |

Com 1e-9, um único atributo "estranho" decide a classe sozinho. Foi isso que derrubou um fold do credit-g para **0,43 de acurácia**. Com 0,01, a penalidade continua grande, mas não absurda.

### 4.4 `predict`: dando nota para cada classe

Para cada cliente e cada classe, a nota é:

```
nota(classe) = log(priori) + soma, em cada atributo, de:
               -½ × [ log(2π × variância) + (valor − média)² / variância ]
```

Esse termo é o **log do sino**. Lendo por partes:

- `(valor − média)² / variância`: o quão longe o valor está do centro, **medido em "larguras de sino"**. Quanto mais longe, mais negativa a nota.
- `log(2π × variância)`: penaliza sinos muito largos (uma classe que "aceita qualquer coisa" não deve ganhar fácil).

**Exemplo à mão** (um atributo só, valores inventados):

| | `good` | `bad` |
|---|---|---|
| priori | 0,70 | 0,30 |
| média / variância da duração | 20 / 100 | 30 / 144 |

Cliente com duração **24**:

| Passo | `good` | `bad` |
|---|---|---|
| desvio² / variância | (24−20)²/100 = 0,16 | (24−30)²/144 = 0,25 |
| log do sino = −½ [log(2π·var) + desvio²/var] | −3,30 | −3,53 |
| + log(priori) | −3,30 + (−0,36) = **−3,66** | −3,53 + (−1,20) = **−4,73** |

A maior nota é **−3,66 → `good`**. Notas negativas são normais: só importa **qual é maior**.

### 4.5 Como o código faz isso sem laços

Em vez de três laços (cliente × classe × atributo), o código monta uma "tabela 3D" de uma vez:

```
diff = X[:, np.newaxis, :] - self.mean        → formato (n, C, d)
```

Leia `diff[i, k, j]` como: **"o desvio do cliente i em relação à média da classe k, no atributo j"**. O `np.newaxis` só abre uma dimensão vazia para o numpy saber que deve comparar **cada cliente com cada classe**.

Depois:

1. `log_likelihood = -0.5 * (...)` aplica a fórmula do sino em todas as células → (n, C, d).
2. `.sum(axis=2)` soma os atributos → (n, C): uma nota por cliente e por classe.
3. `+ np.log(self.prior)` soma o log da priori de cada classe.
4. `np.argmax(scores, axis=1)` pega, em cada linha, a coluna da maior nota (0 ou 1).
5. `self.classes[...]` troca a posição pelo rótulo da classe.

---

## 5. Bayesiano Multivariado (`GaussianBayes`)

### 5.1 A ideia: atributos andam juntos

No mundo real, os atributos se relacionam: empréstimos **longos** costumam ter **valores altos**. O univariado ignora isso. O multivariado modela todos os atributos juntos com uma **normal multivariada**, que é descrita por:

- **vetor de médias μ** (d valores): o "centro" da classe;
- **matriz de covariância Σ** (d × d): a "forma" da nuvem de pontos da classe.

### 5.2 A matriz de covariância Σ, sem mistério

Σ é uma tabela d × d:

- **diagonal:** a variância de cada atributo (o mesmo do univariado);
- **fora da diagonal:** a **covariância** entre dois atributos:
  - positiva → crescem juntos;
  - negativa → um sobe, o outro desce;
  - zero → sem relação linear.

**Exemplo pequeno** (2 atributos, 4 clientes da classe A):

| Cliente | atributo 1 | atributo 2 |
|---|---|---|
| 1 | 1 | 2 |
| 2 | 2 | 4 |
| 3 | 3 | 6 |
| 4 | 2 | 3 |

```
μ = (2,  3,75)

Σ = [ 0,67   1,33 ]     ← variância do atributo 1 | covariância 1-2
    [ 1,33   2,92 ]     ← covariância 2-1         | variância do atributo 2
```

A covariância 1,33 é **positiva**: quando o atributo 1 sobe, o atributo 2 também sobe. O univariado jogaria essa informação fora. Repare também que Σ é **simétrica**: a covariância 1-2 é igual à 2-1.

> **Relação entre os dois modelos:** se todas as covariâncias fossem zero (Σ diagonal), o multivariado viraria exatamente o univariado. O univariado é um **caso particular** do multivariado.

### 5.3 Distância de Mahalanobis: a "régua inteligente"

No multivariado, a pergunta "o cliente está longe do centro da classe?" é respondida pela **distância de Mahalanobis**:

```
D² = (x − μ)ᵀ Σ⁻¹ (x − μ)
```

Ela é uma distância que **leva em conta a forma da nuvem**. Exemplo: uma classe com centro (0, 0), variâncias 1 e covariância 0,8 (atributos fortemente ligados, nuvem alongada na diagonal "/"):

| Ponto | Distância comum (Euclidiana) | Mahalanobis² |
|---|---|---|
| A = (1, 1) | 1,41 | **1,11** |
| B = (1, −1) | 1,41 | **10,0** |

Os dois pontos estão à **mesma distância** em linha reta, mas:

- A segue o padrão da classe ("os dois atributos sobem juntos") → **perto**;
- B quebra o padrão (um sobe, o outro desce) → **muito longe**.

É exatamente isso que o univariado não consegue ver.

### 5.4 A nota de cada classe (função discriminante)

Aplicando o log na densidade normal multivariada:

```
g(classe) = −½ × Mahalanobis²   −   ½ × log|Σ|   +   log(priori)
              (quão longe)          (quão espalhada)     (quão comum)
```

| Termo | Efeito na nota |
|---|---|
| −½ × Mahalanobis² | Quanto mais longe do centro da classe, menor a nota. |
| −½ × log\|Σ\| | \|Σ\| (determinante) mede o "volume" da nuvem. Classes muito espalhadas perdem pontos. |
| + log(priori) | Classes mais frequentes ganham um bônus. |

O termo −d/2 × log(2π) da fórmula foi **omitido**: ele é igual para todas as classes e não muda qual nota é maior.

**Paralelo com o univariado:** é a mesma estrutura, `−½ (distância) − ½ log(espalhamento) + log(priori)`. No univariado a "distância" é atributo por atributo; aqui ela considera as correlações.

### 5.5 `fit`: aprendendo o resumo de cada classe

```
classes = valores distintos de y
I = matriz identidade d × d
para cada classe c:
    Xk = só os clientes da classe c
    priori[c] = nº de clientes da classe / nº total
    média[c]  = média de cada coluna de Xk
    Σ[c]      = np.cov(Xk) + λ × I          ← regularização
depois, uma vez só:
    Σ⁻¹[c]    = inversa de cada Σ
    log|Σ[c]| = log do determinante de cada Σ
```

Detalhes da linha da covariância:

- `np.cov(Xk, rowvar=False)`: por padrão, o `np.cov` espera **um atributo por linha**. Nos nossos dados é **um atributo por coluna**, então `rowvar=False` avisa isso. Ele divide por (n − 1), o estimador amostral.
- `np.atleast_2d(...)`: com um único atributo, `np.cov` devolveria um número solto. Isso garante o formato de tabela d × d.
- `+ self.reg_lambda * identity`: soma λ **só na diagonal** (seção 5.6).

**Por que pré-calcular Σ⁻¹ e log|Σ| no `fit`?** Eles só dependem do treino. Calculando uma vez, o `predict` fica mais rápido.

O que fica guardado no credit-g (C = 2, d = 50 com `drop_first`):

| Nome | Formato | Conteúdo |
|---|---|---|
| `prior` | (2,) | priori de cada classe |
| `mean` | (2, 50) | vetor de médias de cada classe |
| `cov` | (2, 50, 50) | uma Σ por classe |
| `inv_cov` | (2, 50, 50) | uma Σ⁻¹ por classe |
| `log_det_cov` | (2,) | log\|Σ\| de cada classe |

### 5.6 Os problemas numéricos e como foram resolvidos

O multivariado precisa **inverter Σ** e calcular seu **determinante**. Isso falha quando Σ é **singular** (determinante 0, sem inversa), o que acontece quando:

- um atributo é **constante** na classe;
- atributos são **linearmente dependentes**, como colunas one-hot completas: elas **sempre somam 1**, então qualquer uma pode ser calculada a partir das outras;
- há **poucas amostras** na classe em comparação ao número de atributos.

| Problema | Solução no código |
|---|---|
| Colunas one-hot dependentes | `drop_first=True` no leitor ARFF: remove uma categoria por atributo (63 → 50 colunas). A informação não se perde: "todas as outras são 0" já indica a categoria removida. |
| Σ singular ou quase singular | **Regularização Σ + λI**: somar λ na diagonal garante que Σ tenha inversa. |
| Determinante gigante ou minúsculo | `np.linalg.slogdet` calcula **log\|Σ\| direto**, sem calcular \|Σ\| antes. Em 50 dimensões, \|Σ\| pode estourar ou virar 0; o log dele não. |

**Sobre o `slogdet`:** ele devolve dois valores, o **sinal** do determinante e o **log do valor absoluto**. Como Σ + λI é sempre positiva, o sinal é sempre +1 e o código o descarta (`_`).

**Por que λ = 0,1?** Os dados chegam em z-score, então cada atributo tem variância 1. Assim, λ = 0,1 significa "somar 10% da variância de um atributo à diagonal". É pouco viés, mas o suficiente para deixar Σ bem comportada. Com λ = 1e-6, Σ ficava quase singular e os resultados variavam bastante entre os folds.

### 5.7 `predict`: dando nota para cada classe

O laço percorre as **classes** (só 2), e cada passo processa **todos os clientes de uma vez**:

```
para cada classe k:
    diff = X − média[k]                                → (n, d): desvio de cada cliente
    mahalanobis² = soma por linha de (diff @ Σ⁻¹[k]) × diff  → (n,): um número por cliente
    nota[:, k] = −½ × mahalanobis² − ½ × log|Σ[k]| + log(priori[k])
classe prevista = coluna com a maior nota em cada linha
```

**A linha da Mahalanobis, por partes:** a fórmula é (x − μ)ᵀ Σ⁻¹ (x − μ) para **um** cliente. Para fazer de todos de uma vez:

1. `diff @ self.inv_cov[k]`: multiplica cada linha de desvios por Σ⁻¹ → (n, d);
2. `* diff`: multiplica, elemento a elemento, pelos desvios de novo → (n, d);
3. `np.sum(..., axis=1)`: soma cada linha → um número por cliente (n,).

O resultado é o mesmo que fazer a conta cliente por cliente. Isso foi verificado: as previsões ficaram **idênticas** às de uma versão "manual" em dados de teste.

**Por que o univariado não usa laço e este usa?** No multivariado, a tabela 3D teria uma Σ inteira por célula. O laço nas classes (só 2) é simples de ler e não custa quase nada.

---

## 6. Univariado × Multivariado

| | Univariado | Multivariado |
|---|---|---|
| Vê relação entre atributos? | Não (supõe independência) | Sim (covariâncias) |
| O que guarda por classe | d médias + d variâncias | d médias + matriz d × d |
| Parâmetros no credit-g | 2 × (63 + 63) = 252 | 2 × (50 + 1.275) = 2.650 |
| "Distância" usada | atributo por atributo | Mahalanobis |
| Fronteira entre classes | mais simples | quádrica (curvas: elipses, parábolas); também chamado de **QDA** |
| Problema numérico típico | variância zero → `var_smoothing` | Σ singular → `drop_first` + λI |
| Precisa de mais dados? | Não | Sim: mais parâmetros, mais risco de overfitting |

O número 1.275 vem de d(d+1)/2 = 50 × 51 / 2: como Σ é simétrica, só metade dela (mais a diagonal) é informação nova.

---

## 7. Resultados e como interpretá-los

credit-g, 5 folds estratificados, seed 42:

| Classificador | Acurácia | Precisão | Recall | F1-Score |
|---|---|---|---|---|
| Bayesiano Univariado | 0,693 ± 0,023 | 0,671 ± 0,032 | 0,698 ± 0,041 | 0,668 ± 0,029 |
| Bayesiano Multivariado | 0,724 ± 0,025 | 0,677 ± 0,028 | 0,681 ± 0,032 | 0,676 ± 0,028 |
| *Referência: sempre "good"* | *0,700* | *0,350* | *0,500* | *0,412* |

Os tempos de treino e teste dos dois ficam abaixo de 1 milissegundo por fold.

**Como ler:**

- **Compare sempre com a referência.** 70% dos clientes são `good`, então "chutar sempre good" já dá 0,70 de acurácia. O univariado fica perto disso em acurácia, mas os dois são **muito melhores em F1** (0,67 contra 0,41), porque realmente identificam parte dos `bad`.
- **O multivariado ganha** em acurácia e F1. Faz sentido: ele aproveita as correlações entre atributos.
- **Recall:** o univariado tem recall um pouco maior. Atenção: as métricas são **macro**, ou seja, a média do recall de `good` e de `bad`. Um recall macro maior indica que o modelo trata as duas classes de forma mais equilibrada, não que acerta mais uma delas em especial.
- **± desvio:** mostra a estabilidade entre os folds. Valores em torno de 0,02–0,03 indicam resultados estáveis.

**Sobre os hiperparâmetros (`var_smoothing = 0,01` e `λ = 0,1`):** o enunciado não pede ajuste de hiperparâmetros. Os valores foram escolhidos de forma exploratória, e a justificativa é técnica (evitar variâncias e covariâncias quase zero nas colunas one-hot). Se perguntarem, seja transparente: como foram escolhidos olhando os próprios folds, o resultado pode estar levemente otimista. O correto seria uma validação interna ao treino, que ficou fora do escopo.

---

## 8. Perguntas prováveis na apresentação

**Por que usar log?**
Para evitar underflow (produtos de muitos números pequenos viram 0) e trocar multiplicação por soma. A classe vencedora não muda, porque o log preserva a ordem.

**Por que dá para ignorar p(x), a evidência?**
Ela é a mesma para todas as classes. Dividir todas as notas pelo mesmo número não muda qual é a maior.

**Por que "naive"?**
Porque supõe que os atributos são independentes dentro de cada classe, o que quase nunca é verdade. Mesmo assim funciona razoavelmente, porque só precisa acertar **qual** classe é mais provável, não o valor exato da probabilidade.

**O que é a matriz de covariância?**
A diagonal tem a variância de cada atributo; fora dela, o quanto cada par de atributos varia junto. Ela descreve a forma da nuvem de pontos da classe.

**O que é a distância de Mahalanobis?**
Uma distância que considera a escala e a correlação dos atributos. Um ponto que segue o padrão da classe fica "perto", mesmo que em linha reta pareça longe. Se Σ for a identidade, ela vira a distância Euclidiana.

**Por que somar λI?**
Para garantir que Σ tenha inversa. Colunas one-hot, atributos constantes ou poucos dados deixam Σ singular. Somar λ só na diagonal resolve com pouco viés.

**Por que `drop_first=True` só no multivariado?**
As colunas one-hot de um atributo sempre somam 1, o que torna Σ singular. O univariado não inverte matriz nenhuma, então não precisa disso.

**Por que o determinante aparece na fórmula?**
Ele mede o "volume" da distribuição. Sem ele, uma classe muito espalhada "aceitaria" qualquer ponto. O termo −½ log|Σ| equilibra isso.

**Por que `slogdet` e não `det`?**
Em 50 dimensões, o determinante pode ser um número enorme ou minúsculo demais para o computador. `slogdet` dá o log direto, sem passar pelo valor perigoso.

**Por que o multivariado não ganhou de lavada?**
O credit-g tem muitas colunas binárias (one-hot), que não seguem uma curva normal. Além disso, o multivariado estima muito mais parâmetros (2.650 contra 252) com só 800 clientes de treino por fold, o que aumenta o risco de overfitting.

**Usaram alguma biblioteca de machine learning?**
Não. Só numpy para operações matemáticas (média, covariância, inversa, log). A lógica do classificador (priori, verossimilhança, MAP) foi toda implementada à mão.

---

## 9. Resumo para os slides

**Slide: Teorema de Bayes e regra MAP**
- posteriori ∝ priori × verossimilhança
- MAP: escolhe a classe de maior posteriori
- Na prática: log-pontuação = log(priori) + log(verossimilhança)

**Slide: Univariado × Multivariado**
- Univariado: um sino por atributo e por classe; supõe independência; rápido e simples.
- Multivariado: um "sino em d dimensões" por classe; captura correlações (Σ); usa a distância de Mahalanobis.
- Cuidados numéricos: `var_smoothing` (univariado), `drop_first` + Σ + λI + log-determinante (multivariado).
- Resultado: multivariado 0,724 de acurácia e 0,676 de F1; univariado 0,693 e 0,668; referência 0,70 e 0,41.

| Conceito | Em uma frase |
|---|---|
| Priori | Quão comum é cada classe no treino. |
| Verossimilhança | Quão típico o cliente seria naquela classe. |
| MAP | Escolhe a classe de maior pontuação. |
| Log | Troca produto por soma e evita underflow. |
| var_smoothing | ε somado às variâncias para não dividir por zero. |
| Σ | Variâncias na diagonal, covariâncias fora dela. |
| Mahalanobis | Distância que respeita a forma da nuvem da classe. |
| λI | Soma na diagonal de Σ para ela ter inversa. |
| drop_first | Remove uma coluna one-hot por atributo para evitar dependência linear. |
