from dotenv import load_dotenv

load_dotenv()

from src.sync_database import sync_database
from src.utils import tickers

sync_database(tickers)