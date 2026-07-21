# Avaliação do Modelo LSTM

Após o treinamento de cada modelo LSTM, é necessário avaliar sua capacidade de prever corretamente o preço de fechamento das ações. Como o problema é de regressão, são utilizadas métricas que medem o erro entre os valores previstos pelo modelo e os valores reais observados no conjunto de teste.

Além da avaliação do modelo, também é calculado o desempenho de um modelo de referência (*baseline*), denominado **Naive Forecast**, permitindo verificar se a rede neural realmente aprende padrões úteis ou apenas reproduz o comportamento mais simples possível.


# 1. Evitando Data Leakage

Antes do cálculo das métricas, é importante garantir que o processo de treinamento não utilize informações provenientes do futuro.

Em séries temporais, o erro mais comum consiste em ajustar o escalonador (*scaler*) utilizando toda a base de dados. Quando isso acontece, informações estatísticas do conjunto de teste (como mediana e intervalo interquartil, no caso do RobustScaler) acabam sendo utilizadas durante o treinamento, produzindo métricas artificialmente melhores.

O procedimento correto consiste em:

1. Ordenar cronologicamente toda a série temporal;
2. Dividir a base em treino, validação e teste;
3. Ajustar (`fit`) o scaler **apenas no conjunto de treinamento**;
4. Aplicar (`transform`) esse scaler nos conjuntos de validação e teste;
5. Construir as janelas temporais somente após a normalização.

No projeto, esse procedimento é realizado da seguinte forma:

```python
n = len(data)

train_end = int(n * 0.70)
valid_end = int(n * 0.85)

train_df = data.iloc[:train_end]
valid_df = data.iloc[train_end:valid_end]
test_df = data.iloc[valid_end:]

scaler = RobustScaler()
scaler.fit(train_df)

train_scaled = scaler.transform(train_df)
valid_scaled = scaler.transform(valid_df)
test_scaled = scaler.transform(test_df)
```

Esse procedimento elimina o risco de **data leakage**, produzindo uma avaliação mais fiel da capacidade de generalização do modelo.


# 2. Estatísticas Descritivas

Antes da avaliação propriamente dita, algumas estatísticas da série de preços são calculadas.

## Média

Representa o preço médio da ação durante todo o período utilizado.

```text
μ = (1/n) Σ yi
```

onde:

- yi representa cada preço de fechamento;
- n representa o número total de observações.

## Mediana

É o valor central da série quando os preços são ordenados.

A mediana é menos sensível a valores extremos do que a média.

## Desvio padrão

Mede o quanto os preços variam em torno da média.

```text
σ = √( Σ (yi − μ)² / (n−1) )
```

Quanto maior o desvio padrão, maior a volatilidade da ação.

## Coeficiente de Variação (CV)

Como ações possuem preços em escalas diferentes, o desvio padrão isoladamente não permite comparação direta.

Utiliza-se então o coeficiente de variação.

```text
CV = (σ / μ) × 100
```

Esse indicador representa a volatilidade relativa da ação em porcentagem.

# 3. Métricas de avaliação

## MAE (Mean Absolute Error)

O MAE mede o erro absoluto médio entre previsão e valor real.

```text
MAE = (1/n) Σ |yi − ŷi|
```

onde:

- yi é o valor real;
- ŷi é o valor previsto.

Como utiliza valores absolutos, todos os erros possuem o mesmo peso.

Quanto menor o MAE, melhor.

Sua unidade é a mesma do preço da ação (reais).

## RMSE (Root Mean Squared Error)

O RMSE mede a raiz do erro quadrático médio.

```text
RMSE = √( (1/n) Σ (yi − ŷi)² )
```

Por elevar os erros ao quadrado antes da média, erros grandes recebem penalização muito maior.

Por isso, o RMSE é bastante sensível a previsões muito distantes do valor real.

Quanto menor, melhor.

## MAPE (Mean Absolute Percentage Error)

O MAPE mede o erro percentual médio.

```text
MAPE = (100/n) Σ |(yi − ŷi)/yi|
```

Sua principal vantagem é produzir um erro independente da escala dos preços.

Exemplo:

- MAPE = 2%

significa que, em média, as previsões erram aproximadamente 2% do valor real.

## sMAPE (Symmetric Mean Absolute Percentage Error)

O sMAPE é uma versão simétrica do MAPE.

```text
sMAPE = (100/n) Σ
2 |yi − ŷi|
──────────────
|yi| + |ŷi|
```

Sua vantagem é evitar distorções quando os valores reais estão muito próximos de zero.

Em aplicações financeiras, costuma produzir medidas mais estáveis que o MAPE.

## R² (Coeficiente de Determinação)

O coeficiente de determinação mede quanto da variabilidade dos preços é explicada pelo modelo.

```text
R² =
1 −
Σ(yi − ŷi)²
──────────────
Σ(yi − ȳ)²
```

Valores possíveis:

- R² = 1 → previsão perfeita;
- R² = 0 → desempenho equivalente à média dos dados;
- R² < 0 → pior do que simplesmente prever a média.

Quanto mais próximo de 1, melhor.

# 4. Modelo Naive

Além da rede LSTM, foi utilizado um modelo extremamente simples denominado **Naive Forecast**.

Nesse modelo, a previsão para o próximo dia é simplesmente o preço observado no último dia disponível.

Formalmente:

```text
ŷ(t+1) = y(t)
```

No código, isso corresponde a:

```python
naive_scaled = X_test[:, -1, 0]
```

Como cada janela temporal possui os últimos preços observados, o último elemento da janela representa exatamente o fechamento mais recente conhecido.

Após desfazer a normalização, obtém-se a previsão do modelo Naive.

## Por que comparar com o Naive?

Em séries temporais financeiras, o preço de hoje normalmente é muito parecido com o de ontem.

Isso significa que um modelo extremamente simples pode apresentar erros relativamente baixos.

Caso uma rede neural profunda não consiga superar esse modelo, ela provavelmente não está aprendendo padrões relevantes, apenas reproduzindo o comportamento básico da série.

Por esse motivo, o Naive é amplamente utilizado como baseline na literatura de previsão de séries temporais.

## MAE Naive

Corresponde ao erro absoluto médio obtido pelo modelo Naive.

```text
MAE_naive =
(1/n)
Σ |yi − y(t−1)|
```

## RMSE Naive

Erro quadrático médio do modelo Naive.

```text
RMSE_naive =
√(
(1/n)
Σ (yi − y(t−1))²
)
```

## MAPE Naive

Erro percentual médio produzido pelo modelo Naive.

```text
MAPE_naive =
(100/n)
Σ |(yi − y(t−1))/yi|
```

## Skill Score (RMSE)

O Skill Score mede o ganho obtido pelo modelo em relação ao baseline Naive.

Neste projeto foi utilizada a seguinte definição:

```text
Skill =
1 −
RMSEmodelo
────────────
RMSEnaive
```

Interpretação:

- Skill > 0 → modelo melhor que o Naive;
- Skill = 0 → desempenho equivalente;
- Skill < 0 → modelo pior que o Naive.

Quanto maior o Skill Score, maior o ganho obtido pela rede LSTM.

## MAE/Media (%)

Como ações possuem preços muito diferentes, o MAE absoluto nem sempre é facilmente interpretável.

Uma forma de normalizar esse erro consiste em dividi-lo pelo preço médio da ação.

```text
MAE/Media =
(MAE / Média) × 100
```

Esse indicador representa o erro absoluto médio em relação ao preço médio negociado.

## RMSE/Media (%)

Da mesma forma, o RMSE também pode ser normalizado.

```text
RMSE/Media =
(RMSE / Média) × 100
```

Essa métrica facilita a comparação entre ativos com escalas de preço bastante diferentes.

## Melhor que Naive

Por fim, é criada uma variável booleana indicando se o modelo superou ou não o baseline.

```python
skill > 0
```

Se verdadeiro:

- o modelo apresentou RMSE inferior ao Naive.

Caso contrário:

- utilizar apenas o preço do dia anterior produziria previsões mais precisas do que a rede neural.

Essa comparação é particularmente importante em problemas de previsão financeira, nos quais séries altamente autocorrelacionadas tornam o modelo Naive um baseline bastante competitivo.