from dotenv import load_dotenv

load_dotenv()

from src.sync_database import sync_database
from src.generate_predict import generate_predictions, read_table
from src.utils import tickers
from src.setup_logging import WorkflowLogger

# sync_database(tickers)
# generate_predictions(tickers)

logger = WorkflowLogger("daily_sync")

try:

    sync_database(tickers)
    generate_predictions(tickers)

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