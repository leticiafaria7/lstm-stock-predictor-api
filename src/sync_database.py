# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

import pandas as pd
import numpy as np

from src.instances import supabase

from src.get_tables import (
    get_historical_data_assets,
    get_historical_data_tickers,
    get_historical_br_indexes
)

# ----------------------------------------------------------------------------------------------- #
# Configurações
# ----------------------------------------------------------------------------------------------- #

# yfinance usa auto_adjust=True, que reajusta retroativamente o
# histórico de preços (dividendos/JCP) a cada download, relativo à
# data do download. Como a sincronização é incremental, precisamos
# re-sobrescrever uma janela recente para "curar" esse ajuste sempre
# que um provento é declarado. Mantida menor que antes (30 em vez de
# 90 dias) para reduzir a exposição a respostas ruins/parciais do
# yfinance a cada sync.
LOOKBACK_DAYS_HEALING = 30

# Variação diária máxima considerada plausível para um ajuste de
# dividendo/JCP normal. Qualquer novo valor que se desvie do valor já
# salvo mais que isso é tratado como suspeito (provável dado ruim
# vindo do yfinance) e NÃO sobrescreve o que já está no banco.
MAX_PLAUSIBLE_CHANGE_PCT = 0.08  # 8%

# ----------------------------------------------------------------------------------------------- #
# Funções auxiliares
# ----------------------------------------------------------------------------------------------- #

def table_is_empty(table_name):

    response = (
        supabase
        .table(table_name)
        .select("Date")
        .limit(1)
        .execute()
    )

    return len(response.data) == 0


def last_date(table_name):

    response = (
        supabase
        .table(table_name)
        .select("Date")
        .order("Date", desc=True)
        .limit(1)
        .execute()
    )

    return response.data[0]["Date"]


def healed_start_date(table_name, lookback_days=LOOKBACK_DAYS_HEALING):

    ultima = pd.to_datetime(last_date(table_name))
    inicio = ultima - pd.Timedelta(days=lookback_days)

    return inicio.strftime("%Y-%m-%d")


def fetch_existing_rows(table_name, start_date, key_columns):
    """
    Busca do Supabase os registros já salvos a partir de `start_date`,
    para servir de referência na validação (evita comparar contra
    nada e sobrescrever tudo às cegas).
    """

    response = (
        supabase
        .table(table_name)
        .select("*")
        .gte("Date", start_date)
        .execute()
    )

    existing = pd.DataFrame(response.data)

    if existing.empty:
        return existing

    existing["Date"] = existing["Date"].astype(str)

    return existing


def validate_tickers_df(new_df, table_name="historical_tickers"):
    """
    Compara cada linha nova (Date, Ticker) com o valor já salvo no
    Supabase, se existir. Se a variação do 'Close' for maior que
    MAX_PLAUSIBLE_CHANGE_PCT, a linha é considerada suspeita (provável
    resposta ruim/parcial do yfinance) e é DESCARTADA do upsert -
    preferimos manter o dado antigo a corromper o histórico.
    """

    if new_df.empty:
        return new_df

    new_df = new_df.copy()
    new_df["Date"] = new_df["Date"].astype(str)

    start_date = new_df["Date"].min()
    existing = fetch_existing_rows(table_name, start_date, ["Date", "Ticker"])

    if existing.empty:
        return new_df

    merged = new_df.merge(
        existing[["Date", "Ticker", "Close"]],
        on=["Date", "Ticker"],
        how="left",
        suffixes=("", "_old")
    )

    tem_referencia = merged["Close_old"].notna()

    variacao = (
        (merged["Close"] - merged["Close_old"]).abs()
        / merged["Close_old"].abs()
    )

    suspeita = tem_referencia & (
        merged["Close"].isna()
        | (merged["Close"] <= 0)
        | (variacao > MAX_PLAUSIBLE_CHANGE_PCT)
    )

    if suspeita.any():

        descartadas = merged.loc[
            suspeita, ["Date", "Ticker", "Close", "Close_old"]
        ]

        print(
            "[sync_database] Linhas suspeitas descartadas da atualização "
            f"(variação > {MAX_PLAUSIBLE_CHANGE_PCT:.0%} ou valor inválido):\n"
            f"{descartadas.to_string(index=False)}"
        )

    return new_df.loc[~suspeita].reset_index(drop=True)


def upsert_dataframe(df, table_name, on_conflict):

    if df.empty:
        return

    df = df.copy()

    # remove colunas que não existem no Supabase
    df = df.drop(columns=["Adj Close"], errors="ignore")

    df["Date"] = df["Date"].astype(str)

    # elimina infinitos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # converte todo NaN em None
    records = (
        df.astype(object)
        .where(pd.notnull(df), None)
        .to_dict("records")
    )

    supabase.table(table_name).upsert(
        records,
        on_conflict=on_conflict
    ).execute()

# ----------------------------------------------------------------------------------------------- #
# Função que popula as tabelas, e se já estiver populada, preenche ate a data atual
# ----------------------------------------------------------------------------------------------- #

def sync_database(tickers):

    # Histórico ações -----------------------------------------------------

    if table_is_empty("historical_tickers"):

        df = get_historical_data_tickers(tickers)
        upsert_dataframe(df, "historical_tickers", "Date,Ticker")

    else:

        inicio = healed_start_date("historical_tickers")

        df = get_historical_data_tickers(tickers, start=inicio)
        df = validate_tickers_df(df, "historical_tickers")

        upsert_dataframe(df, "historical_tickers", "Date,Ticker")

    # Histórico ativos ------------------------------------------------------

    if table_is_empty("historical_assets"):

        df = get_historical_data_assets()
        upsert_dataframe(df, "historical_assets", "Date")

    else:

        inicio = last_date("historical_assets")

        df = get_historical_data_assets(start=inicio)
        upsert_dataframe(df, "historical_assets", "Date")

    # Histórico índices --------------------------------------------------

    if table_is_empty("historical_indexes"):

        df = get_historical_br_indexes()
        upsert_dataframe(df, "historical_indexes", "Date")

        print("Carga inicial concluída.")

    else:

        inicio = last_date("historical_indexes")

        df = get_historical_br_indexes(start=inicio)
        upsert_dataframe(df, "historical_indexes", "Date")

        print("Atualização concluída.")
