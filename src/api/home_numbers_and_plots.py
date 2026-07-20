import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from flask import render_template
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from src.instances import bp, supabase
from src.generate_predict import get_analitical_table

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "train_model/models")

TICKERS = [
    "ABEV3",
    "RENT3",
    "LREN3",
    "SMFT3"
]

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

def load_table(table_name):

    data = []
    offset = 0

    while True:

        response = (
            supabase
            .table(table_name)
            .select("*")
            .range(offset, offset + 999)
            .execute()
        )

        if not response.data:
            break

        data.extend(response.data)

        if len(response.data) < 1000:
            break

        offset += 1000

    return pd.DataFrame(data)


def load_tables():

    hist = load_table("historical_tickers")
    assets = load_table("historical_assets")
    indexes = load_table("historical_indexes")

    hist["Date"] = pd.to_datetime(hist["Date"])
    assets["Date"] = pd.to_datetime(assets["Date"])
    indexes["Date"] = pd.to_datetime(indexes["Date"])

    # print(f'#### TAMANHO DOS DATASETS {hist.shape} {assets.shape} {indexes.shape}')

    return hist, assets, indexes


def smape(real, pred):

    return 100*np.mean(
        2*np.abs(pred-real)/(np.abs(real)+np.abs(pred)+1e-8)
    )


def build_lstm_sequences(window_size, scaled):

    X=[]
    y=[]

    for i in range(window_size,len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i,0])

    return np.array(X),np.array(y)


def inverse_close(values, scaler, n_features):

    dummy=np.zeros((len(values),n_features))
    dummy[:,0]=values.ravel()

    return scaler.inverse_transform(dummy)[:,0]


def load_metadata(ticker):

    return joblib.load(
        os.path.join(
            MODELS_DIR,
            f"metadata_{ticker}.pkl"
        )
    )


def load_model_and_scaler(ticker):

    model=load_model(
        os.path.join(
            MODELS_DIR,
            f"lstm_{ticker}.keras"
        )
    )

    scaler=joblib.load(
        os.path.join(
            MODELS_DIR,
            f"scaler_{ticker}.pkl"
        )
    )

    metadata=load_metadata(ticker)

    return model,scaler,metadata


def create_dataset(ticker, hist, assets, indexes):

    df=get_analitical_table(
        ticker,
        hist,
        assets,
        indexes
    )

    return (
        df
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

def rebuild_predictions(ticker, hist, assets, indexes):

    model, scaler, metadata = load_model_and_scaler(ticker)

    features = metadata["features"]
    window_size = metadata["window_size"]

    df = create_dataset(ticker, hist, assets, indexes)

    data = df[features].copy()

    n = len(data)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_df = data.iloc[:train_end].copy()
    valid_df = data.iloc[train_end:valid_end].copy()
    test_df = data.iloc[valid_end:].copy()

    scaler.fit(train_df)

    train_scaled = scaler.transform(train_df)
    valid_scaled = scaler.transform(valid_df)
    test_scaled = scaler.transform(test_df)

    _, _ = build_lstm_sequences(window_size, train_scaled)
    _, _ = build_lstm_sequences(window_size, valid_scaled)

    X_test, y_test = build_lstm_sequences(window_size, test_scaled)

    pred_scaled = model.predict(X_test, verbose=0)

    naive_scaled = X_test[:, -1, 0]

    pred = inverse_close(pred_scaled, scaler, len(features))
    real = inverse_close(y_test, scaler, len(features))
    naive = inverse_close(naive_scaled, scaler, len(features))

    dates = (
        df.iloc[valid_end + window_size:]
        .date
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )

    metrics = {
        "MAE": round(mean_absolute_error(real, pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(real, pred)), 3),
        "MAPE": round(mean_absolute_percentage_error(real, pred) * 100, 2),
        "SMAPE": round(smape(real, pred), 2)
    }

    return {
        "real": real,
        "pred": pred,
        "naive": naive,
        "dates": dates,
        "metrics": metrics
    }


# paleta extraída da identidade visual do dashboard (mesmas cores do home.html)
CHART_COLORS = {
    "real": "#432818",     # espresso
    "pred": "#99582a",     # rust
    "naive": "#bb9457",    # tan
    "grid": "rgba(67,40,24,0.08)",
    "text": "#6b5240"
}


def build_prediction_chart(result, ticker):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result["dates"],
            y=result["real"],
            mode="lines",
            name="Real",
            line=dict(width=2.6, color=CHART_COLORS["real"])
        )
    )

    fig.add_trace(
        go.Scatter(
            x=result["dates"],
            y=result["pred"],
            mode="lines",
            name="LSTM",
            line=dict(width=2.6, color=CHART_COLORS["pred"])
        )
    )

    fig.add_trace(
        go.Scatter(
            x=result["dates"],
            y=result["naive"],
            mode="lines",
            name="Naive",
            line=dict(width=1.8, color=CHART_COLORS["naive"], dash="dot")
        )
    )

    fig.update_layout(
        title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Roboto, sans-serif", size=12, color=CHART_COLORS["text"]),
        hovermode="x unified",
        height=420,
        margin=dict(l=10, r=10, t=15, b=10),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0,
            font=dict(size=11)
        ),
        xaxis=dict(
            showgrid=False,
            rangeslider_visible=False,
            tickfont=dict(size=10, color=CHART_COLORS["text"])
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=CHART_COLORS["grid"],
            zeroline=False,
            tickfont=dict(size=10, color=CHART_COLORS["text"])
        )
    )

    return fig.to_json()


def load_all_results():

    hist, assets, indexes = load_tables()

    metrics = {}
    charts = {}

    for ticker in TICKERS:

        result = rebuild_predictions(
            ticker,
            hist,
            assets,
            indexes
        )

        metrics[ticker] = result["metrics"]
        charts[ticker] = build_prediction_chart(result, ticker)

    return metrics, charts


def build_company_cards():

    cards = []

    for ticker in TICKERS:

        info = COMPANIES[ticker]

        cards.append({
            "ticker": ticker,
            "empresa": info["empresa"],
            "setor": info["setor"],
            "subsetor": info["subsetor"],
            "segmento": info["segmento"],
            "inicio": info["inicio"],
            "fim": info["fim"]
        })

    return cards


def build_feature_cards():

    return [
        {
            "name": feature,
            "description": FEATURE_DESCRIPTIONS[feature]
        }
        for feature in FEATURE_DESCRIPTIONS
    ]


def build_endpoint_cards():

    return [
        {
            "method": "POST",
            "path": "/api/v1/predict/<ticker>",
            "description": "Retorna a previsão do preço de fechamento para D+1."
        },
        {
            "method": "GET",
            "path": "/api/v1/models",
            "description": "Lista os modelos e tickers suportados."
        },
        {
            "method": "GET",
            "path": "/api/v1/models/<ticker>/metrics",
            "description": "Retorna as métricas do modelo treinado."
        },
        {
            "method": "GET",
            "path": "/health",
            "description": "Endpoint para health check do serviço."
        },
        {
            "method": "GET",
            "path": "/metrics",
            "description": "Métricas Prometheus para monitoramento."
        }
    ]


def build_sidebar():

    return [
        {
            "id": "sobre",
            "title": "Projeto"
        },
        {
            "id": "acoes",
            "title": "Ações"
        },
        {
            "id": "features",
            "title": "Features"
        },
        {
            "id": "modelos",
            "title": "Modelos"
        },
        {
            "id": "api",
            "title": "API"
        }
    ]


def build_dashboard():

    metrics, charts = load_all_results()

    company_cards = build_company_cards()
    feature_cards = build_feature_cards()
    endpoint_cards = build_endpoint_cards()
    sidebar = build_sidebar()

    dashboard = {
        "sidebar": sidebar,
        "companies": company_cards,
        "features": feature_cards,
        "endpoints": endpoint_cards,
        "metrics": metrics,
        "charts": charts,
        "default_ticker": TICKERS[0],
        "tickers": TICKERS
    }

    return dashboard


def serialize_dashboard(dashboard):

    return {
        "sidebar": dashboard["sidebar"],
        "companies": dashboard["companies"],
        "features": dashboard["features"],
        "endpoints": dashboard["endpoints"],
        "tickers": dashboard["tickers"],
        "default_ticker": dashboard["default_ticker"],
        "metrics": dashboard["metrics"],
        "charts": dashboard["charts"],
        "metrics_json": json.dumps(dashboard["metrics"]),
        "charts_json": json.dumps(dashboard["charts"]),
        "features_json": json.dumps(dashboard["features"])
    }

@bp.route("/", methods=["GET"])
def home():

    dashboard = build_dashboard()
    context = serialize_dashboard(dashboard)

    return render_template(
        "home.html",
        sidebar=context["sidebar"],
        companies=context["companies"],
        empresas_dict=COMPANIES,
        features=context["features"],
        endpoints=context["endpoints"],
        tickers=context["tickers"],
        default_ticker=context["default_ticker"],
        metrics=context["metrics"],
        charts=context["charts"],
        metrics_json=context["metrics_json"],
        charts_json=context["charts_json"],
        features_json=context["features_json"],
        github_url="https://github.com/leticiafaria7",
        swagger_url="/apidocs/",
        project_title="Modelo LSTM para prever preço de fechamento de ações",
        project_description="API Flask para previsão diária de preço de fechamento utilizando modelos LSTM treinados individualmente para cada ativo. Os dados são sincronizados diariamente com o Supabase, processados automaticamente e disponibilizados por meio de endpoints REST documentados via Swagger."
    )