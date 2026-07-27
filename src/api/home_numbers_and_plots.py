# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

import json

from flask import render_template

from src.instances import bp, supabase
from src.utils import COMPANIES

# ----------------------------------------------------------------------------------------------- #
# Leitura do cache já pronto (gravado 1x por dia pelo sync.py)
# ----------------------------------------------------------------------------------------------- #

def load_dashboard_cache():

    response = (
        supabase
        .table("dashboard_cache")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "dashboard_cache está vazia. Rode o sync.py para popular a tabela."
        )

    return response.data[0]


def serialize_dashboard(cache):

    return {
        "sidebar": cache["sidebar"],
        "companies": cache["companies"],
        "features": cache["features"],
        "endpoints": cache["endpoints"],
        "tickers": cache["tickers"],
        "default_ticker": cache["default_ticker"],
        "metrics": cache["metrics"],
        "charts": cache["charts"],
        "metrics_json": json.dumps(cache["metrics"]),
        "charts_json": json.dumps(cache["charts"]),
        "features_json": json.dumps(cache["features"])
    }

# ----------------------------------------------------------------------------------------------- #
# Rota Home
# ----------------------------------------------------------------------------------------------- #

@bp.route("/", methods=["GET"])
def home():

    cache = load_dashboard_cache()
    context = serialize_dashboard(cache)

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
        github_url="https://github.com/leticiafaria7/lstm-stock-predictor-api",
        swagger_url="/apidocs/",
        project_title="Modelo LSTM para prever preço de fechamento de ações",
        project_description="API Flask para previsão diária de preço de fechamento utilizando modelos LSTM treinados individualmente para cada ativo. Os dados são sincronizados diariamente com o Supabase, processados automaticamente e disponibilizados por meio de endpoints REST documentados via Swagger."
    )
