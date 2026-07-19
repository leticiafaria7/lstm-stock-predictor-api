
# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

from pathlib import Path
import joblib

from flask import jsonify
from src.instances import supabase, bp

MODELS_DIR = Path("train_model/models")

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
        description: Código do ativo para gerar a previsão.

    responses:
      200:
        description: Previsão encontrada para o ticker informado.
        schema:
          type: object
          properties:
            Ticker:
              type: string
              example: "ABEV3"
              description: Código do ativo previsto.
            Date:
              type: string
              format: date
              example: "2026-07-20"
              description: Data de referência da previsão.
            Close:
              type: number
              example: 12.45
              description: Último preço de fechamento observado.
            Predict:
              type: number
              example: 12.62
              description: Preço de fechamento previsto para o próximo pregão.

      404:
        description: Modelo ou previsão não encontrados.
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
              example: "Erro ao consultar previsão"
    """

    file = MODELS_DIR / f"metadata_{ticker}.pkl"

    if not file.exists():
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
        return jsonify({"error": "Previsão não encontrada"}), 404

    return jsonify(r.data[0])

# ----------------------------------------------------------------------------------------------- #
# Métricas para Prometheus
# ----------------------------------------------------------------------------------------------- #

@bp.route("/metrics", methods=["GET"])
def metrics():
    """
    Retorna métricas da aplicação no formato Prometheus.

    ---
    tags:
      - Monitoring

    responses:
      200:
        description: Métricas da API no formato Prometheus.
        content:
          text/plain:
            example: |
              # HELP api_predictions_total Number of predictions
              # TYPE api_predictions_total gauge
              api_predictions_total 120

              # HELP api_database_up Database status
              # TYPE api_database_up gauge
              api_database_up 1

        schema:
          type: string
          description: Métricas de monitoramento para coleta pelo Prometheus.

      500:
        description: Erro interno ao gerar as métricas.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Erro ao gerar métricas"
    """

    rows = len(read_table("predictions"))

    text = f"""
# HELP api_predictions_total Number of predictions
# TYPE api_predictions_total gauge
api_predictions_total {rows}

# HELP api_database_up Database status
# TYPE api_database_up gauge
api_database_up {1 if check_database() else 0}
"""

    return text, 200, {"Content-Type": "text/plain"}
