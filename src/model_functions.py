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
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2

# ==============================================================================
# Orquestração do modelo
# ==============================================================================

def build_lstm_sequences(window_size, scaled_data):
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        # últimos WINDOW_SIZE dias
        X.append(scaled_data[i-window_size:i])
        # target = close do dia atual
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)

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

def generate_lstm_model(units_1, dropout, units_2, learning_rate, recurrent_dropout, l2_lambda, n_dense, dense_units, optimizer_name, X_train):

    if optimizer_name == "Adam":
        optimizer = Adam(learning_rate)

    elif optimizer_name == "Nadam":
        optimizer = Nadam(learning_rate)

    else:
        optimizer = RMSprop(learning_rate)
    
    model = Sequential()

    model.add(LSTM(units=units_1,
                   return_sequences=True,
                   recurrent_dropout=recurrent_dropout, 
                   kernel_regularizer=l2(l2_lambda),
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units_2,
                   recurrent_dropout=recurrent_dropout,
                   kernel_regularizer=l2(l2_lambda)))
    model.add(Dropout(dropout))
    for _ in range(n_dense):
        model.add(Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(1))

    model.compile(
        optimizer=optimizer,
        loss="mse", 
        metrics=["mae"])

    return model

def get_study_optuna(train_scaled, valid_scaled, test_scaled, n_trials, ticker):

    def objective(trial):

        # Hiperparâmetros
        units_1 = trial.suggest_int("units_1", 32, 256, step=32)
        units_2 = trial.suggest_int("units_2", 16, 128, step=16)
        dropout = trial.suggest_float("dropout", 0.10, 0.50)
        recurrent_dropout = trial.suggest_float("recurrent_dropout", 0.0, 0.40)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        l2_lambda = trial.suggest_float("l2", 1e-6, 1e-2, log=True)
        dense_units = trial.suggest_categorical("dense_units", [8, 16, 32, 64])
        n_dense = trial.suggest_int("n_dense", 1, 2)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Nadam", "RMSprop"])
        window_size = trial.suggest_categorical("window_size", [30, 60, 90, 120, 180, 360])

        # Modelo
        X_train, y_train = build_lstm_sequences(window_size, train_scaled)
        X_valid, y_valid = build_lstm_sequences(window_size, valid_scaled)

        model = generate_lstm_model(units_1, dropout, units_2, learning_rate, recurrent_dropout, l2_lambda, n_dense, dense_units, optimizer_name, X_train)

        model.fit(
            X_train, 
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=100,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
            verbose=0
        )

        val_loss, val_mae = model.evaluate(X_valid, y_valid, verbose=0)

        return val_loss
    
    study = optuna.create_study(direction="minimize", study_name=f"LSTM Stock Prediction | {ticker}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("=" * 60)
    print(f"Melhores hiperparâmetros | Ticker = {ticker}")
    print("=" * 60)

    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    print("\nMelhor validation loss:")
    print(study.best_value)

    return study