# ==============================================================================
# Imports
# ==============================================================================

import numpy as np

import optuna

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==============================================================================
# Orquestração do modelo
# ==============================================================================

def build_lstm_sequences(window_size, scaled_data):

    X = []
    y = []

    for i in range(window_size, len(scaled_data)):

        # últimos WINDOW_SIZE dias
        X.append(scaled_data[i-window_size:i])

        # target = close do dia atual
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    print(X.shape)
    print(y.shape)

    return X, y

def split_temporal(X, y, train_size = 0.70, valid_size = 0.15):

    n = len(X)

    train_end = int(n * train_size)
    valid_end = int(n * (train_size + valid_size))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_valid = X[train_end:valid_end]
    y_valid = y[train_end:valid_end]

    X_test = X[valid_end:]
    y_test = y[valid_end:]

    print(f"Treino     : {X_train.shape}")
    print(f"Validação  : {X_valid.shape}")
    print(f"Teste      : {X_test.shape}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def generate_lstm_model(units_1, dropout, units_2, learning_rate, X_train):
    
    model = Sequential()

    model.add(LSTM(units=units_1,
                   return_sequences=True, 
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units_2))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate = learning_rate),
                loss="mse", 
                metrics=["mae"])

    return model


def get_study_optuna(w, scaled_data, n_trials, ticker):

    window_size = w

    X, y = build_lstm_sequences(window_size, scaled_data)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_temporal(X, y)

    def objective(trial):

        # Hiperparâmetros
        units_1 = trial.suggest_int("units_1", 32, 128, step=32)
        units_2 = trial.suggest_int("units_2", 16, 64, step=16)
        dropout = trial.suggest_float("dropout", 0.10, 0.50)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # Modelo
        model = generate_lstm_model(units_1, dropout, units_2, learning_rate, X_train)

        history = model.fit(
            X_train, 
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=100,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
            verbose=0
        )

        return min(history.history["val_loss"])
    
    study = optuna.create_study(direction="minimize", study_name=f"LSTM Stock Prediction | window_size = {w}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("=" * 60)
    print(f"Melhores hiperparâmetros | Ticker = {ticker} | window_size = {w}")
    print("=" * 60)

    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    print("\nMelhor validation loss:")
    print(study.best_value)

    return study