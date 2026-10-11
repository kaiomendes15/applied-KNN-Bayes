# Naive Bayes — Guia de Primeiro Contato

Este guia explica o que é o Naive Bayes, como ele decide a classe de uma amostra e como transformar essa lógica em código. Os exemplos usam o dataset do projeto, o `credit-g`, que classifica clientes de crédito como **`good`** (bom pagador) ou **`bad`** (mau pagador).

---

## 1. A ideia em uma frase

> O Naive Bayes olha para um cliente novo e pergunta: **"com qual grupo de clientes do treino ele mais se parece, levando em conta também o tamanho de cada grupo?"**

Ele não decora regras. Ele **resume cada classe** com alguns números (médias e variâncias) e depois compara o cliente novo com esses resumos.

---

## 2. Os ingredientes

O método vem do **Teorema de Bayes**, que combina duas informações:

| Ingrediente | Pergunta que responde | Exemplo no credit-g |
|---|---|---|
| **Priori** — P(classe) | "Antes de olhar o cliente, qual classe é mais comum?" | 70% dos clientes são `good` e 30% são `bad` |
| **Verossimilhança** — p(x \| classe) | "Se o cliente fosse desta classe, quão normal seria ele ter estes atributos?" | Um empréstimo de 48 meses é comum entre os `bad`? |

Multiplicando as duas, obtemos uma **pontuação** para cada classe:

```
pontuação(classe) = priori(classe) × verossimilhança(x | classe)
```

O resultado é proporcional à **posteriori**, P(classe | x), que é a probabilidade de o cliente ser daquela classe **dado o que observamos nele**.

**Regra de decisão:** a classe com a **maior pontuação** vence. Essa regra se chama **MAP** (Máximo A Posteriori).

> Não precisamos dividir pela evidência p(x), que aparece no denominador do Teorema de Bayes, porque ela é igual para todas as classes. Ela não muda qual pontuação é a maior.

---

## 3. Por que "Naive" (ingênuo)?

Um cliente tem vários atributos: duração do empréstimo, valor, idade… Calcular a chance de **todos juntos** acontecerem é difícil, porque eles se influenciam.

O Naive Bayes faz uma **simplificação ingênua**: ele supõe que, dentro de cada classe, **os atributos não têm relação entre si**. Com isso, a verossimilhança vira uma simples multiplicação:

```
verossimilhança(x | classe) = p(duração | classe) × p(valor | classe) × p(idade | classe) × ...
```

Na vida real os atributos se relacionam (empréstimos longos costumam ter valores altos), então a suposição é "ingênua". Mesmo assim, o método costuma funcionar bem.

---

## 4. Por que média e variância?

Para calcular p(duração | classe), precisamos de uma "forma" para os dados. Usamos a **distribuição normal**, a curva em formato de sino. Por isso a versão do projeto se chama **Naive Bayes Gaussiano**.

Uma curva normal fica totalmente definida por dois números:

- **Média (μ):** onde fica o centro do sino. É o valor típico.
- **Variância (σ²):** quão largo é o sino, ou seja, o quanto os valores se espalham.

Quanto mais perto do centro o valor do cliente estiver, maior a densidade e, portanto, maior a verossimilhança.

A fórmula (está no enunciado) é:

$$p(x) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right]$$

**Importante:** cada classe tem **seu próprio sino para cada atributo**. A média de duração dos `good` é diferente da média de duração dos `bad`.

---

## 5. Exemplo à mão

Resumos (aproximados) calculados no credit-g:

| Classe | Priori | Duração: média / desvio | Valor: média / desvio |
|---|---|---|---|
| `good` | 0,70 | 19 meses / 11 | 3.000 / 2.400 |
| `bad` | 0,30 | 25 meses / 13 | 3.900 / 3.500 |

**Cliente novo:** empréstimo de **48 meses** no valor de **9.000**.

| Classe | p(duração) | p(valor) | Pontuação = priori × p(duração) × p(valor) |
|---|---|---|---|
| `good` | 0,00112 | 0,0000073 | 0,70 × 0,00112 × 0,0000073 ≈ **5,7 × 10⁻⁹** |
| `bad` | 0,00642 | 0,0000394 | 0,30 × 0,00642 × 0,0000394 ≈ **7,6 × 10⁻⁸** |

**Resultado: `bad`.** Mesmo com a priori favorecendo `good` (70%), o cliente está muito mais perto do perfil típico dos maus pagadores. Os atributos "venceram" a priori.

Repare em como os números ficaram minúsculos com **apenas 2 atributos**. Isso leva ao próximo ponto.

---

## 6. O truque do logaritmo

No projeto são cerca de **48 atributos**. Multiplicar 48 números pequenos gera um valor tão próximo de zero que o computador o arredonda para **0** (*underflow*). Aí todas as classes empatam em zero e o classificador não consegue decidir.

A solução é trabalhar com **logaritmo**:

- O log transforma **multiplicação em soma**: log(a × b) = log(a) + log(b).
- Somar números negativos moderados não causa underflow.
- O log **preserva a ordem**: se A > B, então log(A) > log(B). Então a classe vencedora continua a mesma.

```
log-pontuação(classe) = log(priori) + soma, para cada atributo, de log(p(atributo | classe))
```

Aplicando o log na fórmula da normal, cada termo da soma fica:

```
log p(x) = -0.5 × [ log(2π × variância) + (x - média)² / variância ]
```

Assim não é preciso calcular `exp` nem raiz quadrada.

---

## 7. Como aplicar: o passo a passo

O classificador tem duas etapas, seguindo o contrato do projeto (`fit` e `predict`).

### `fit(X, y)` — aprender os resumos de cada classe

```
para cada classe k:
    Xk = linhas de X que pertencem à classe k
    priori[k]    = quantidade de linhas de Xk / total de linhas
    media[k]     = média de cada coluna de Xk
    variancia[k] = variância de cada coluna de Xk + suavização
```

O que fica guardado (com C classes e d atributos):

| Atributo | Formato | No credit-g |
|---|---|---|
| `classes` | `(C,)` | `(2,)` |
| `prior` | `(C,)` | `(2,)` |
| `mean` | `(C, d)` | `(2, 48)` |
| `var` | `(C, d)` | `(2, 48)` |

A **suavização** (`var_smoothing`) é um número pequeno somado à variância. Ela evita **divisão por zero** quando uma coluna tem sempre o mesmo valor dentro de uma classe.

### `predict(X)` — comparar cada amostra com os resumos

```
para cada amostra x:
    para cada classe k:
        pontuacao[k] = log(priori[k]) + soma dos log p(x_j | classe k)
    previsão = classe com a maior pontuação
```

### Dicas de numpy

| Tarefa | Ferramenta |
|---|---|
| Descobrir as classes | `np.unique(y)` |
| Selecionar linhas de uma classe | `X[y == k]` |
| Média e variância por coluna | `.mean(axis=0)` e `.var(axis=0)` |
| Escolher a maior pontuação | `np.argmax(..., axis=1)` |

> **Desafio de eficiência:** dá para calcular as pontuações de **todas as amostras ao mesmo tempo**, sem laço sobre as amostras, usando *broadcasting* do numpy. Comece com laços para entender e depois tente vetorizar.

---

## 8. Cuidados no credit-g

- **Colunas constantes:** `purpose=vacation` e `personal_status=female single` são sempre zero no dataset, porque nenhuma amostra usa essas categorias. A variância delas é zero. Remova-as ou confie na suavização.
- **Classes desbalanceadas (70/30):** um classificador que sempre responde `good` já acerta 70%. Por isso a acurácia sozinha engana. Olhe também **F1, precisão e recall**.
- **Normalização:** o Naive Bayes Gaussiano não precisa dela, porque cada atributo tem sua própria média e variância. Ela é obrigatória no kNN, mas aqui é opcional.

---

## 9. Resumo

| Conceito | Em uma frase |
|---|---|
| Priori | Quão comum cada classe é no treino. |
| Verossimilhança | Quão típico o cliente seria dentro daquela classe. |
| Posteriori | A chance da classe depois de olhar o cliente; é o que comparamos. |
| Naive | Supõe que os atributos são independentes dentro de cada classe. |
| Gaussiano | Cada atributo é modelado por um sino (média + variância). |
| Log | Troca produto por soma para evitar números que viram zero. |
| MAP | Escolhe a classe de maior pontuação. |

**Pontos fortes:** é muito rápido para treinar e prever, simples e funciona com poucos dados.
**Limitações:** a suposição de independência raramente é verdadeira, e nem todo atributo segue uma curva normal (as colunas one-hot, por exemplo, só valem 0 ou 1).
