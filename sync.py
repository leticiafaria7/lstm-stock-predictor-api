from dotenv import load_dotenv

load_dotenv()

from src.sync_database import sync_database
from src.generate_predict import generate_predictions
from src.utils import tickers

sync_database(tickers)
generate_predictions(tickers)