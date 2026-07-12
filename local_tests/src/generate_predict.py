
from local_tests.src.get_tables import get_historical_data_tickers, get_historical_data_assets, get_historical_br_indexes, get_analitical_table
from src.utils import features

from tensorflow.keras.models import load_model
import joblib
import numpy as np

def generate_predict_ticker(ticker, data_final):

    ultima_data = ultima_data_disponivel()
    start = ultima_data + 1

    historical_data_tickers = get_historical_data_tickers(tickers, save_folder_path, start = start, end = data_final)
    historical_data_assets = get_historical_data_assets(save_folder_path, start = start, end = data_final)
    historical_br_indexes = get_historical_br_indexes(save_folder_path, start = start)

    analitical_table = get_analitical_table(save_folder_path, ticker)

    model = load_model(f"models/lstm_{ticker}.keras")
    scaler = joblib.load(f"scalers/scaler_{ticker}.pkl")

    novos_dados = analitical_table[features]

    X_scaled = scaler.transform(novos_dados)
    pred_scaled = model.predict(X_scaled)

    dummy_pred = np.zeros((len(pred_scaled), len(features)))
    dummy_pred[:, 0] = pred_scaled.ravel()
    pred = scaler.inverse_transform(dummy_pred)[:, 0]

    return pred