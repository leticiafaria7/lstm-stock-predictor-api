
# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

from pathlib import Path
import joblib
import time
import pandas as pd

from flask import jsonify, Response
from src.instances import supabase, bp

MODELS_DIR = Path("train_model/models")

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.monitoring import (
    DATABASE_UP,
    PREDICTIONS_ROWS,
    SUPPORTED_MODELS,
    MODEL_MAE,
    MODEL_RMSE,
    MODEL_MAPE,
    WORKFLOW_DURATION,
    WORKFLOW_LAST_SUCCESS,
    WORKFLOW_FAILURES,
    PREDICTION_REQUESTS,
    PREDICTION_ERRORS,
    PREDICTION_DURATION
)

# ----------------------------------------------------------------------------------------------- #
# Funções auxiliares
# ----------------------------------------------------------------------------------------------- #

def read_table(table):

    data = []
    offset = 0

    while True:

        r = (
            supabase.table(table)
            .select("*")
            .range(offset, offset + 999)
            .execute()
        )

        if not r.data:
            break

        data.extend(r.data)
        offset += 1000

    return data

def check_database():

    try:

        (
            supabase.table("historical_tickers")
            .select("Ticker")
            .limit(1)
            .execute()
        )

        return True

    except Exception:

        return False

# ----------------------------------------------------------------------------------------------- #
# Health - Verificar status da API e conectividade com os dados
# ----------------------------------------------------------------------------------------------- #

@bp.route("/health", methods=["GET"])
def health():
    """
    ---
    tags:
      - Health
    summary: Verifica o estado da API
    description: Retorna o status operacional da API, conexão com o banco de dados e disponibilidade dos dados de previsão.
    responses:
      200:
        description: API funcionando corretamente
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: ok
                api:
                  type: string
                  example: running
                database:
                  type: string
                  example: ok
                data_loaded:
                  type: boolean
                  example: true
                rows:
                  type: integer
                  example: 4
      503:
        description: Banco de dados indisponível
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: ok
                api:
                  type: string
                  example: running
                database:
                  type: string
                  example: error
                data_loaded:
                  type: boolean
                  example: false
                rows:
                  type: integer
                  example: 0
      500:
        description: Erro interno do servidor
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: Internal server error
    """

    try:
        rows = len(read_table("predictions"))

        status = {
            "status": "ok",
            "api": "running",
            "database": "ok" if check_database() else "error",
            "data_loaded": rows > 0,
            "rows": rows
        }

        code = 200 if status["database"] == "ok" else 503
        return jsonify(status), code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------------------------------------- #
# Modelos disponíveis
# ----------------------------------------------------------------------------------------------- #

@bp.route("/api/v1/models", methods=["GET"])
def models():
    """
    Lista os modelos LSTM disponíveis na API.

    ---
    tags:
      - Models

    responses:
      200:
        description: Lista de modelos disponíveis.
        schema:
          type: array
          items:
            type: object
            properties:
              ticker:
                type: string
                example: "ABEV3"
                description: Código do ativo associado ao modelo.
              window_size:
                type: integer
                example: 30
                description: Quantidade de períodos históricos utilizados como entrada do modelo LSTM.
              features:
                type: integer
                example: 21
                description: Quantidade de variáveis utilizadas no treinamento do modelo.

      500:
        description: Erro interno ao carregar os metadados dos modelos.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Erro ao carregar modelos"
    """

    modelos = []

    for file in MODELS_DIR.glob("metadata_*.pkl"):

        meta = joblib.load(file)

        modelos.append(
            {
                "ticker": meta["ticker"],
                "window_size": meta["window_size"],
                "features": len(meta["features"])
            }
        )

    return jsonify(modelos)

# ----------------------------------------------------------------------------------------------- #
# Métricas de avaliação do modelo
# ----------------------------------------------------------------------------------------------- #

@bp.route("/api/v1/models/<ticker>/metrics", methods=["GET"])
def model_metrics(ticker):
    """
    Retorna as métricas de avaliação do modelo LSTM de um ticker específico.

    ---
    tags:
      - Models

    parameters:
      - name: ticker
        in: path
        required: true
        type: string
        example: "ABEV3"
        description: Código do ativo associado ao modelo.

    responses:
      200:
        description: Métricas de avaliação do modelo.
        schema:
          type: object
          properties:
            MAE:
              type: number
              example: 0.2075
              description: Erro absoluto médio do modelo.
            RMSE:
              type: number
              example: 0.2976
              description: Raiz do erro quadrático médio do modelo.
            MAPE:
              type: number
              example: 1.5338
              description: Erro percentual absoluto médio do modelo.
            R2:
              type: number
              example: 0.9681
              description: Coeficiente de determinação do modelo.
            Skill_RMSE:
              type: number
              example: -0.3795
              description: Comparação do desempenho do modelo contra o baseline naive.

      404:
        description: Modelo não encontrado para o ticker informado.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Ticker não suportado"

      500:
        description: Erro interno ao carregar as métricas do modelo.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Erro ao carregar métricas"
    """

    file = MODELS_DIR / f"metadata_{ticker}.pkl"

    if not file.exists():
        return jsonify({"error": "Ticker não suportado"}), 404

    meta = joblib.load(file)

    return jsonify(meta["metrics"])

# ----------------------------------------------------------------------------------------------- #
# Predict
# ----------------------------------------------------------------------------------------------- #

@bp.route("/api/v1/predict/<ticker>", methods=["POST"])
def predict(ticker):
    """
    Retorna a previsão do preço de fechamento para o próximo pregão de um ticker.

    ---
    tags:
      - Prediction

    parameters:
      - name: ticker
        in: path
        required: true
        type: string
        example: "ABEV3"
        description: Código do ativo para consultar a previsão.

    responses:
      200:
        description: Previsão encontrada com sucesso.
        schema:
          type: object
          properties:
            Ticker:
              type: string
              example: "ABEV3"
              description: Código do ativo.
            Date:
              type: string
              format: date
              example: "2026-07-20"
              description: Data de referência da previsão.
            Close:
              type: number
              format: float
              example: 12.45
              description: Último preço de fechamento disponível.
            Predict_D1:
              type: number
              format: float
              example: 12.62
              description: Previsão do preço de fechamento para o próximo pregão.

      404:
        description: Ticker não suportado ou previsão inexistente.
        schema:
          type: object
          properties:
            error:
              type: string
              examples:
                ticker:
                  value: "Ticker não suportado"
                prediction:
                  value: "Previsão não encontrada"

      500:
        description: Erro interno ao consultar a previsão.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Erro interno do servidor"
    """

    start = time.perf_counter()

    PREDICTION_REQUESTS.labels(
        ticker=ticker
    ).inc()

    try:

        file = MODELS_DIR / f"metadata_{ticker}.pkl"

        if not file.exists():

            PREDICTION_ERRORS.labels(
                ticker=ticker
            ).inc()

            return jsonify({"error": "Ticker não suportado"}), 404

        r = (
            supabase.table("predictions")
            .select("*")
            .eq("Ticker", ticker)
            .order("Date", desc=True)
            .limit(1)
            .execute()
        )

        if len(r.data) == 0:

            PREDICTION_ERRORS.labels(
                ticker=ticker
            ).inc()

            return jsonify({"error": "Previsão não encontrada"}), 404

        return jsonify(r.data[0])

    finally:

        PREDICTION_DURATION.labels(
            ticker=ticker
        ).observe(
            time.perf_counter() - start
        )

# ----------------------------------------------------------------------------------------------- #
# Métricas para Prometheus
# ----------------------------------------------------------------------------------------------- #

@bp.route("/metrics", methods=["GET"])
def metrics():

    """
    Retorna as métricas da aplicação no formato Prometheus.

    ---
    tags:
      - Monitoring

    produces:
      - text/plain

    responses:
      200:
        description: Métricas da API para coleta pelo Prometheus.
        schema:
          type: string
          example: |
            # HELP http_requests_total Total HTTP requests
            # TYPE http_requests_total counter
            http_requests_total{endpoint="/health",method="GET",status="200"} 10

            # HELP prediction_requests_total Prediction requests
            # TYPE prediction_requests_total counter
            prediction_requests_total{ticker="ABEV3"} 15

            # HELP database_up Database status
            # TYPE database_up gauge
            database_up 1

            # HELP model_mae Model MAE
            # TYPE model_mae gauge
            model_mae{ticker="ABEV3"} 0.208

    description: |
      Endpoint utilizado pelo Prometheus para coleta periódica das métricas da API.

      As principais métricas disponibilizadas incluem:

      - http_requests_total
      - http_request_duration_seconds
      - http_request_errors_total
      - prediction_requests_total
      - prediction_latency_seconds
      - prediction_errors_total
      - database_up
      - predictions_rows
      - supported_models
      - model_mae
      - model_rmse
      - model_mape
      - workflow_duration_seconds
      - workflow_last_success_timestamp
      - workflow_failures_total
      """

    DATABASE_UP.set(1 if check_database() else 0)

    PREDICTIONS_ROWS.set(len(read_table("predictions")))

    model_files = list(MODELS_DIR.glob("metadata_*.pkl"))
    SUPPORTED_MODELS.set(len(model_files))

    for file in model_files:

        meta = joblib.load(file)
        ticker = meta["ticker"]

        MODEL_MAE.labels(ticker=ticker).set(meta["metrics"]["mae"])
        MODEL_RMSE.labels(ticker=ticker).set(meta["metrics"]["rmse"])
        MODEL_MAPE.labels(ticker=ticker).set(meta["metrics"]["mape"])

    logs = read_table("workflow_logs")

    logs = pd.DataFrame(read_table("workflow_logs"))

    if not logs.empty:

        logs = logs.sort_values("created_at")

        last = logs.iloc[-1]

        WORKFLOW_DURATION.set(last["duration_seconds"])

        if last["status"] == "success":
            WORKFLOW_LAST_SUCCESS.set(pd.Timestamp(last["created_at"]).timestamp())

        WORKFLOW_FAILURES.set(int((logs["status"] == "error").sum()))

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)