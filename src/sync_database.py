
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

    # records = df.to_dict("records")

    supabase.table(table_name).upsert(
        records,
        on_conflict=on_conflict
    ).execute()
    
# ----------------------------------------------------------------------------------------------- #
# Função que popula as tabelas, e se já estiver populada, preenche ate a data atual
# ----------------------------------------------------------------------------------------------- #

def sync_database(tickers):

    # Histórico ações ----------------------------------------------------

    if table_is_empty("historical_tickers"):

        df = get_historical_data_tickers(tickers)
        upsert_dataframe(df, "historical_tickers", "Date,Ticker")

    else:

        inicio = last_date("historical_tickers")

        df = get_historical_data_tickers(tickers, start=inicio)
        upsert_dataframe(df, "historical_tickers", "Date,Ticker")

    # Histórico ativos ---------------------------------------------------

    if table_is_empty("historical_assets"):

        df = get_historical_data_assets()
        upsert_dataframe(df,"historical_assets", "Date")

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
