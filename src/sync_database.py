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

# yfinance usa auto_adjust=True, que reajusta retroativamente TODO o
# histórico de preços (dividendos/JCP/splits) a cada download, relativo
# à data do download. Como a sincronização é incremental, isso cria uma
# quebra de escala entre o preço "antigo" já salvo (ajustado até a data
# do sync anterior) e o preço "novo" (ajustado até hoje) sempre que uma
# data-ex de provento acontece entre dois syncs. Para curar isso,
# rebaixamos o ponto de partida em alguns dias e deixamos o upsert
# sobrescrever os registros recentes já existentes com o fator de
# ajuste atualizado.
LOOKBACK_DAYS_HEALING = 90

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
    """
    Retorna a última data salva menos `lookback_days`, para que o
    próximo fetch reescreva (upsert) a janela recente com o fator de
    ajuste de proventos mais atual, em vez de só buscar dados novos.
    """

    ultima = pd.to_datetime(last_date(table_name))
    inicio = ultima - pd.Timedelta(days=lookback_days)

    return inicio.strftime("%Y-%m-%d")


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
    # (afetado por dividendos/JCP -> precisa da janela de "cura")

    if table_is_empty("historical_tickers"):

        df = get_historical_data_tickers(tickers)
        upsert_dataframe(df, "historical_tickers", "Date,Ticker")

    else:

        inicio = healed_start_date("historical_tickers")

        df = get_historical_data_tickers(tickers, start=inicio)
        upsert_dataframe(df, "historical_tickers", "Date,Ticker")

    # Histórico ativos ------------------------------------------------------
    # (índices/câmbio não sofrem ajuste retroativo por proventos, mantém incremental)

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
