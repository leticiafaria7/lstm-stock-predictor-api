
# ----------------------------------------------------------------------------------------------- #
# Tickers e features
# ----------------------------------------------------------------------------------------------- #

tickers = [
    "ABEV3",
    "RENT3",
    "LREN3",
    "SMFT3"
]

features = [
    'close', 'high', 'low', 'open', 'volume',
    'close_dolar', 'close_ibovespa', 'close_sp_500', 'selic', 'ipca',
    'ma20', 'ma50', 'bb_upper', 'bb_lower', 'rsi_wilder', 'macd',
    'macd_signal', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos'
]

TICKERS = tickers

CHART_COLORS = {
    "real": "#432818",     # espresso
    "pred": "#99582a",     # rust
    "naive": "#bb9457",    # tan
    "grid": "rgba(67,40,24,0.08)",
    "text": "#6b5240"
}

COMPANIES = {
    "ABEV3":{
        "empresa":"Ambev S/A",
        "setor":"Consumo não Cíclico",
        "subsetor":"Bebidas",
        "segmento":"Cervejas e Refrigerantes",
        "inicio":"2016-07-01",
        "fim":"2026-05-29"
    },
    "RENT3":{
        "empresa":"Localiza",
        "setor":"Consumo Cíclico",
        "subsetor":"Diversos",
        "segmento":"Aluguel de carros",
        "inicio":"2016-07-01",
        "fim":"2026-05-29"
    },
    "LREN3":{
        "empresa":"Lojas Renner",
        "setor":"Consumo Cíclico",
        "subsetor":"Comércio Varejista",
        "segmento":"Tecidos, Vestuário e Calçados",
        "inicio":"2016-07-01",
        "fim":"2026-05-29"
    },
    "SMFT3":{
        "empresa":"Smart Fit",
        "setor":"Consumo Cíclico",
        "subsetor":"Viagens e Lazer",
        "segmento":"Atividades Esportivas",
        "inicio":"2021-07-14",
        "fim":"2026-05-29"
    }
}


# ----------------------------------------------------------------------------------------------- #
# Descrição das features
# ----------------------------------------------------------------------------------------------- #

rsi_text = """
O RSI (Relative Strength Index) mede a intensidade dos movimentos recentes de alta e baixa do ativo, variando entre 0 e 100.
Valores abaixo de 30 costumam indicar sobrevenda, enquanto valores acima de 70 sugerem sobrecompra.
O RSI Wilder é uma versão tradicional proposta por J. Welles Wilder, calculada utilizando médias móveis exponenciais (EMA), produzindo um indicador mais suave.
Fórmula: RSI = 100 - (100 / (1 + (Média dos ganhos / Média das perdas)))
"""

sp_500 = """
Fechamento do S&P500 no dia de negociação.
S&P 500, abreviação de Standard & Poor's 500, ou simplesmente S&P, 
trata-se de um índice composto por quinhentos ativos cotados nas bolsas de NYSE ou NASDAQ, 
qualificados devido ao seu tamanho de mercado, sua liquidez e sua representação de grupo industrial.
"""

ibovespa_text = """
Fechamento do Ibovespa no dia de negociação.
O índice Ibovespa o principal indicador da B3 (Brasil, Bolsa, Balcão), bolsa de valores oficial do Brasil, que reúne ações de maior volume negociado.
A composição do Ibovespa é reavaliada a cada 4 meses, mas o peso de cada ação muda diariamente com base na oscilação dos preços.
"""

macd_text = """
O MACD (Moving Average Convergence Divergence) é um indicador de momentum baseado na diferença entre duas médias móveis exponenciais (EMA), sendo amplamente utilizado para identificar mudanças de tendência.
A média móvel exponencial (Exponential Moving Average — EMA), por sua vez, é uma variação da média móvel que atribui maior peso aos preços mais recentes e menor peso aos mais antigos,
tornando-se mais sensível às mudanças de tendência do mercado do que a média móvel simples (SMA).
No caso do modelo, foi utilizado um MACD diferença entre a EMA de 12 períodos e a EMA de 26 períodos.
"""

def codificacao_ciclica(componente, var):
    return f"""
Componente {componente} da codificação cíclica do {var}.
Variáveis como dia da semana e mês possuem natureza cíclica, isto é, após o último valor do ciclo ocorre um retorno ao primeiro
(sexta-feira é seguida por segunda-feira no conjunto de pregões, e dezembro é seguido por janeiro).
Representá-las apenas como valores inteiros poderia induzir o modelo a interpretar que existe uma distância muito grande entre o último e o primeiro elemento do ciclo.
Para preservar essa característica, foi utilizada uma codificação baseada nas funções seno e cosseno, que projeta cada categoria em um ponto sobre uma circunferência unitária.
A utilização conjunta das componentes seno e cosseno permite que a LSTM capture padrões sazonais sem introduzir descontinuidades artificiais entre categorias consecutivas do ciclo.
"""


FEATURE_DESCRIPTIONS = {
    "close":"Preço de fechamento da ação no dia de negociação.",
    "high":"Maior preço negociado durante o pregão.",
    "low":"Menor preço negociado durante o pregão.",
    "open":"Preço de abertura da ação no dia de negociação.",
    "volume":"Quantidade de ações negociadas durante o pregão, utilizada como indicador de liquidez e intensidade das negociações.",
    "close_dolar":"Cotação de fechamento do dólar no dia de negociação.",
    "close_ibovespa":ibovespa_text,
    "close_sp_500":sp_500,
    "selic":"Taxa básica de juros da economia brasileira, utilizada como indicador do custo do crédito e da política monetária.",
    "ipca":"Índice Nacional de Preços ao Consumidor Amplo (IPCA), utilizado como principal indicador oficial da inflação brasileira.",
    "ma20":"Média móvel simples calculada sobre os últimos 20 pregões | Fórmula: MA20 = (1/20) × Σ Close(t-i), i = 0,...,19",
    "ma50":"Média móvel simples calculada sobre os últimos 50 pregões | Fórmula: MA20 = (1/50) × Σ Close(t-i), i = 0,...,49",
    "bb_upper":"As Bandas de Bollinger medem a volatilidade do ativo utilizando uma média móvel de 20 períodos e dois desvios padrão. bb_upper é a banda superior, calculada pela fórmula BB_upper = MA20 + 2σ",
    "bb_lower":"As Bandas de Bollinger medem a volatilidade do ativo utilizando uma média móvel de 20 períodos e dois desvios padrão. bb_lower é a banda inferior, calculada pela fórmula BB_upper = MA20 + 2σ",
    "rsi_wilder":rsi_text,
    "macd":macd_text,
    "macd_signal":"Linha de sinal do MACD. É a média móvel exponencial de 9 períodos aplicada ao MACD.",
    "weekday_sin":codificacao_ciclica('seno', 'dia da semana'),
    "weekday_cos":codificacao_ciclica('cosseno', 'dia da semana'),
    "month_sin":codificacao_ciclica('seno', 'mês'),
    "month_cos":codificacao_ciclica('cosseno', 'mês')
}


