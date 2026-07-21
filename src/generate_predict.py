# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from src.instances import supabase
from src.get_tables import get_analitical_table

# ----------------------------------------------------------------------------------------------- #
# Gerar predições
# ----------------------------------------------------------------------------------------------- #

def read_table(table_name, batch_size=1000):

    offset = 0
    data = []

    while True:

        response = (
            supabase.table(table_name)
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
        )

        if not response.data:
            break

        data.extend(response.data)
        offset += batch_size

    return pd.DataFrame(data)


def generate_predictions(tickers):

    hist = read_table("historical_tickers")
    assets = read_table("historical_assets")
    indexes = read_table("historical_indexes")

    hist["Date"] = pd.to_datetime(hist["Date"])
    assets["Date"] = pd.to_datetime(assets["Date"])
    indexes["Date"] = pd.to_datetime(indexes["Date"])

    predictions = []

    for ticker in tickers:

        df = get_analitical_table(
            ticker=ticker,
            df_hist_tickers=hist,
            df_ativos=assets,
            df_br_indexes=indexes
        )

        metadata = joblib.load(f"train_model/models/metadata_{ticker}.pkl")
        scaler = joblib.load(f"train_model/models/scaler_{ticker}.pkl")
        model = load_model(f"train_model/models/lstm_{ticker}.keras", compile = False)

        features = metadata["features"]
        window = metadata["window_size"]

        df = df.dropna().reset_index(drop=True)

        X = scaler.transform(df[features])

        X_pred = X[-window:]
        X_pred = np.expand_dims(X_pred, axis=0)

        pred_scaled = model.predict(X_pred, verbose=0)[0, 0]

        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = pred_scaled

        pred = scaler.inverse_transform(dummy)[0, 0]

        predictions.append(
            {
                "Date": (
                    df.iloc[-1]["date"]
                    .strftime("%Y-%m-%d")
                ),
                "Ticker": ticker,
                "Close": float(
                    df.iloc[-1]["close"]
                ),
                "Predict_D1": float(pred)
            }
        )

    supabase.table("predictions").upsert(
        predictions,
        on_conflict="Date,Ticker"
    ).execute()

    return pd.DataFrame(predictions)
