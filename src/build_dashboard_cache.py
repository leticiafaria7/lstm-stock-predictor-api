# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from src.instances import supabase
from src.generate_predict import get_analitical_table

from src.utils import TICKERS, COMPANIES, FEATURE_DESCRIPTIONS, CHART_COLORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "train_model/models")

# Tamanho FIXO (em dias úteis) da janela usada para o backtest exibido
# no gráfico. Antes era uma % do tamanho da tabela (que cresce todo
# dia), o que fazia o "teste" mudar de composição a cada sync e as
# métricas oscilarem sem o modelo ter mudado. Com uma janela fixa, o
# teste de hoje é quase o mesmo de ontem (só desliza 1 dia).
TEST_WINDOW_DAYS = 90

# ----------------------------------------------------------------------------------------------- #
# Leitura das tabelas históricas
# ----------------------------------------------------------------------------------------------- #

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

    return hist, assets, indexes


def load_latest_prediction(ticker):
    """
    Busca em 'predictions' a última linha (Date mais recente) gravada
    para o ticker por generate_predictions() no mesmo sync. É esse
    valor que vira o último ponto do gráfico, garantindo que o predict
    exibido seja idêntico ao armazenado.
    """

    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("Ticker", ticker)
        .order("Date", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]

# ----------------------------------------------------------------------------------------------- #
# Funções auxiliares de modelagem / métricas
# ----------------------------------------------------------------------------------------------- #

def smape(real, pred):

    return 100*np.mean(
        2*np.abs(pred-real)/(np.abs(real)+np.abs(pred)+1e-8)
    )


def build_lstm_sequences(window_size, scaled):

    X = []
    y = []

    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i, 0])

    return np.array(X), np.array(y)


def inverse_close(values, scaler, n_features):

    dummy = np.zeros((len(values), n_features))
    dummy[:, 0] = values.ravel()

    return scaler.inverse_transform(dummy)[:, 0]


def load_metadata(ticker):

    return joblib.load(
        os.path.join(MODELS_DIR, f"metadata_{ticker}.pkl")
    )


def load_model_and_scaler(ticker):

    model = load_model(
        os.path.join(MODELS_DIR, f"lstm_{ticker}.keras")
    )

    scaler = joblib.load(
        os.path.join(MODELS_DIR, f"scaler_{ticker}.pkl")
    )

    metadata = load_metadata(ticker)

    return model, scaler, metadata


def create_dataset(ticker, hist, assets, indexes):

    df = get_analitical_table(
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


def rebuild_predictions(ticker, hist, assets, indexes, test_window_days=TEST_WINDOW_DAYS):

    model, scaler, metadata = load_model_and_scaler(ticker)

    features = metadata["features"]
    window_size = metadata["window_size"]

    df = create_dataset(ticker, hist, assets, indexes)

    data = df[features].copy()

    n = len(data)

    # Janela de teste FIXA (não proporcional a n): pega os últimos
    # `test_window_days` pontos, com `window_size` dias extras de
    # "aquecimento" antes deles para montar as sequências do LSTM.
    test_start = max(0, n - test_window_days - window_size)
    test_df = data.iloc[test_start:]

    # Transforma com o scaler já treinado (nunca re-fit em produção)
    test_scaled = scaler.transform(test_df)

    X_test, y_test = build_lstm_sequences(window_size, test_scaled)

    pred_scaled = model.predict(X_test, verbose=0)

    naive_scaled = X_test[:, -1, 0]

    pred = inverse_close(pred_scaled, scaler, len(features))
    real = inverse_close(y_test, scaler, len(features))
    naive = inverse_close(naive_scaled, scaler, len(features))

    dates = (
        df.iloc[test_start + window_size:]
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

    # Último ponto: o predict de D+1 já gravado em 'predictions' pelo
    # generate_predictions() (roda antes, no mesmo sync). Não é
    # recalculado aqui, só lido, para o gráfico bater exatamente com
    # o valor armazenado.
    forecast = load_latest_prediction(ticker)

    return {
        "real": real,
        "pred": pred,
        "naive": naive,
        "dates": dates,
        "metrics": metrics,
        "forecast": forecast
    }


def build_prediction_chart(result, ticker):

    dates = list(result["dates"])
    real = list(result["real"])
    pred = list(result["pred"])
    naive = list(result["naive"])

    forecast = result.get("forecast")

    # Acrescenta o ponto de D+1 (ainda sem valor real) usando o mesmo
    # número gravado na tabela 'predictions'.
    if forecast is not None and forecast["Date"] not in dates:

        dates.append(forecast["Date"])
        pred.append(forecast["Predict_D1"])
        real.append(None)
        naive.append(forecast["Close"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=real,
            mode="lines",
            name="Real",
            line=dict(width=2.6, color=CHART_COLORS["real"]),
            connectgaps=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pred,
            mode="lines",
            name="LSTM",
            line=dict(width=2.6, color=CHART_COLORS["pred"])
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=naive,
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

# ----------------------------------------------------------------------------------------------- #
# Conteúdo estático do dashboard (não depende dos modelos)
# ----------------------------------------------------------------------------------------------- #

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
        {"id": "sobre", "title": "Projeto"},
        {"id": "acoes", "title": "Ações"},
        {"id": "features", "title": "Features"},
        {"id": "modelos", "title": "Modelos"},
        {"id": "api", "title": "API"}
    ]

# ----------------------------------------------------------------------------------------------- #
# Monta o dashboard completo
# ----------------------------------------------------------------------------------------------- #

def build_dashboard():

    metrics, charts = load_all_results()

    dashboard = {
        "sidebar": build_sidebar(),
        "companies": build_company_cards(),
        "features": build_feature_cards(),
        "endpoints": build_endpoint_cards(),
        "default_ticker": TICKERS[0],
        "tickers": TICKERS,
        "metrics": metrics,
        "charts": charts
    }

    return dashboard

# ----------------------------------------------------------------------------------------------- #
# Grava o dashboard pronto no Supabase (tabela dashboard_cache, linha única id=1)
# ----------------------------------------------------------------------------------------------- #

def save_dashboard_cache(dashboard):

    record = {
        "id": 1,
        "sidebar": dashboard["sidebar"],
        "companies": dashboard["companies"],
        "features": dashboard["features"],
        "endpoints": dashboard["endpoints"],
        "tickers": dashboard["tickers"],
        "default_ticker": dashboard["default_ticker"],
        "metrics": dashboard["metrics"],
        "charts": dashboard["charts"]
    }

    supabase.table("dashboard_cache").upsert(
        record,
        on_conflict="id"
    ).execute()


def build_and_cache_dashboard():
    """
    Função de entrada, pensada para ser chamada dentro do sync.py,
    DEPOIS de generate_predictions(tickers) (para 'predictions' já
    estar atualizada com o D+1 do dia).
    Reprocessa métricas/gráficos uma única vez por dia e grava o
    resultado pronto (JSON) no Supabase.
    """

    dashboard = build_dashboard()
    save_dashboard_cache(dashboard)

    return dashboard
