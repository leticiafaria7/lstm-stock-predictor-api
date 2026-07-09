# Variáveis preditoras de fechamento de ações

O modelo utiliza variáveis relacionadas ao ativo analisado, indicadores macroeconômicos, índices de mercado e indicadores técnicos de análise financeira. O objetivo é fornecer à LSTM não apenas o histórico de preços da ação, mas também informações sobre o contexto econômico e o comportamento recente do mercado.

> As variáveis taxadas foram removidas por serem variáveis derivadas - o LSTM é capaz de aprender padrões a partir de variáveis brutas

- **close:** preço de fechamento da ação no dia de negociação.
- **high:** maior preço negociado durante o pregão.
- **low:** menor preço negociado durante o pregão.
- **open:** preço de abertura da ação.
- **volume:** quantidade de ações negociadas durante o pregão, utilizada como indicador de liquidez e intensidade das negociações.
- ~~**amplitude_pct:** amplitude percentual do pregão, representando a variação entre o maior e o menor preço do dia.~~

  ```text
  Amplitude (%) = ((High - Low) / Open) × 100
  ```

- ~~**var_dia_pct:** variação percentual entre o preço de abertura e o fechamento do mesmo dia.~~

  ```text
  Variação (%) = ((Close - Open) / Open) × 100
  ```

## Indicadores de mercado

Essas variáveis fornecem ao modelo informações sobre o comportamento de mercados relevantes que podem influenciar o preço da ação analisada.

- **close_dolar:** cotação de fechamento do dólar.

- **close_ibovespa:** fechamento diário do índice Ibovespa.

- **close_sp_500:** fechamento diário do índice S&P 500.

- ~~**amplitude_pct_dolar:** amplitude percentual diária da cotação do dólar.~~

- ~~**amplitude_pct_ibovespa:** amplitude percentual diária do Ibovespa.~~

- ~~**amplitude_pct_sp_500:** amplitude percentual diária do S&P 500.~~

- ~~**var_dia_pct_dolar:** variação percentual diária da cotação do dólar.~~

- ~~**var_dia_pct_ibovespa:** variação percentual diária do Ibovespa.~~

- ~~**var_dia_pct_sp_500:** variação percentual diária do S&P 500.~~

## Indicadores macroeconômicos

- **selic:** taxa básica de juros da economia brasileira, utilizada como indicador do custo do crédito e da política monetária.

- **ipca:** Índice Nacional de Preços ao Consumidor Amplo (IPCA), utilizado como principal indicador oficial da inflação brasileira.

## Médias móveis

As médias móveis suavizam oscilações de curto prazo, facilitando a identificação da tendência predominante do mercado.

- **ma20:** média móvel simples calculada sobre os últimos 20 pregões.

  ```text
  MA20 = (1/20) × Σ Close(t-i), i = 0,...,19
  ```

- **ma50:** média móvel simples calculada sobre os últimos 50 pregões.

  ```text
  MA50 = (1/50) × Σ Close(t-i), i = 0,...,49
  ```

## Bandas de Bollinger

As Bandas de Bollinger medem a volatilidade do ativo utilizando uma média móvel de 20 períodos e dois desvios padrão.

- ~~**bb_middle:** média móvel simples de 20 períodos.~~

  ```text
  BB_middle = MA20
  ```

- **bb_upper:** banda superior.

  ```text
  BB_upper = MA20 + 2σ
  ```

- **bb_lower:** banda inferior.

  ```text
  BB_lower = MA20 - 2σ
  ```

onde `σ` representa o desvio padrão dos últimos 20 períodos.

## Relative Strength Index (RSI)

O RSI mede a intensidade dos movimentos recentes de alta e baixa do ativo, variando entre 0 e 100. Valores abaixo de 30 costumam indicar sobrevenda, enquanto valores acima de 70 sugerem sobrecompra.

- ~~**rsi:** RSI calculado utilizando médias móveis simples (SMA).~~

- **rsi_wilder:** versão tradicional proposta por J. Welles Wilder, calculada utilizando médias móveis exponenciais (EMA), produzindo um indicador mais suave.

O cálculo do RSI é dado por:

```text
RS = Média dos ganhos / Média das perdas
```

```text
RSI = 100 - (100 / (1 + RS))
```

## Moving Average Convergence Divergence (MACD)

O MACD é um indicador de momentum baseado na diferença entre duas médias móveis exponenciais (EMA), sendo amplamente utilizado para identificar mudanças de tendência.

A média móvel exponencial (*Exponential Moving Average* — EMA), por sua vez, é uma variação da média móvel que atribui maior peso aos preços mais recentes e menor peso aos mais antigos, tornando-se mais sensível às mudanças de tendência do mercado do que a média móvel simples (SMA). Seu cálculo utiliza um fator de suavização `α`, definido por:

```text
α = 2 / (N + 1)
```

onde `N` representa o número de períodos considerados.

A EMA é então atualizada recursivamente por:

```text
EMA_t = α × Close_t + (1 - α) × EMA_(t-1)
```

combinando o preço atual com a média exponencial do período anterior.

- **macd:** diferença entre a EMA de 12 períodos e a EMA de 26 períodos.

  ```text
  MACD = EMA12 - EMA26
  ```

- **macd_signal:** média móvel exponencial de 9 períodos aplicada ao MACD.

  ```text
  Signal = EMA9(MACD)
  ```

- ~~**macd_hist:** histograma do MACD, representando a diferença entre o MACD e sua linha de sinal.~~

  ```text
  Histogram = MACD - Signal
  ```

## Codificação cíclica de variáveis temporais

Variáveis como dia da semana e mês possuem natureza cíclica, isto é, após o último valor do ciclo ocorre um retorno ao primeiro (sexta-feira é seguida por segunda-feira no conjunto de pregões, e dezembro é seguido por janeiro). Representá-las apenas como valores inteiros poderia induzir o modelo a interpretar que existe uma distância muito grande entre o último e o primeiro elemento do ciclo. Para preservar essa característica, foi utilizada uma codificação baseada nas funções seno e cosseno, que projeta cada categoria em um ponto sobre uma circunferência unitária.

- **weekday_sin:** componente seno da codificação cíclica do dia da semana, considerando apenas os cinco dias úteis (segunda a sexta).

  ```text
  weekday_sin = sin(2π × weekday / 5)
  ```

- **weekday_cos:** componente cosseno da codificação cíclica do dia da semana.

  ```text
  weekday_cos = cos(2π × weekday / 5)
  ```

onde `weekday ∈ {0,1,2,3,4}` representa segunda-feira até sexta-feira.

- **month_sin:** componente seno da codificação cíclica do mês do ano.

  ```text
  month_sin = sin(2π × (month - 1) / 12)
  ```

- **month_cos:** componente cosseno da codificação cíclica do mês do ano.

  ```text
  month_cos = cos(2π × (month - 1) / 12)
  ```

onde `month ∈ {1,...,12}` representa o mês da data da observação. A utilização conjunta das componentes seno e cosseno permite que a LSTM capture padrões sazonais sem introduzir descontinuidades artificiais entre categorias consecutivas do ciclo.