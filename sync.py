# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

from dotenv import load_dotenv
load_dotenv()

import tempfile
import yfinance as yf

yf.set_tz_cache_location(tempfile.mkdtemp())

from src.sync_database import sync_database
from src.generate_predict import generate_predictions, read_table
from src.build_dashboard_cache import build_and_cache_dashboard
from src.utils import tickers
from src.setup_logging import WorkflowLogger

logger = WorkflowLogger("daily_sync")

# ----------------------------------------------------------------------------------------------- #
# Sincronizar tabelas
# ----------------------------------------------------------------------------------------------- #

try:

    sync_database(tickers)
    generate_predictions(tickers)

    # Reprocessa métricas/gráficos uma única vez por dia e grava o
    # resultado pronto no Supabase, para a API não precisar carregar
    # os modelos LSTM a cada request.
    build_and_cache_dashboard()

    logger.save(
        status="success",
        message="Sincronização concluída com sucesso.",
        rows_historical_tickers=len(read_table("historical_tickers")),
        rows_historical_assets=len(read_table("historical_assets")),
        rows_historical_indexes=len(read_table("historical_indexes")),
        rows_predictions=len(read_table("predictions"))
    )

except Exception as e:

    logger.save(
        status="error",
        message=str(e),
        rows_historical_tickers=len(read_table("historical_tickers")),
        rows_historical_assets=len(read_table("historical_assets")),
        rows_historical_indexes=len(read_table("historical_indexes")),
        rows_predictions=len(read_table("predictions"))
    )

    raise

