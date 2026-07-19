
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