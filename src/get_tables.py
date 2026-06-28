# ---------------------------------------------------------------------------------- #
# Imports
# ---------------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
import plotly.express as px
from datetime import date
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format

# ---------------------------------------------------------------------------------- #
# Variáveis
# ---------------------------------------------------------------------------------- #

tickers = ['RENT3', 'LREN3', 'NATU3', 'SMFT3', 'MULT3', 'VBBR3', 'ABEV3']

sp_500 = """
S&P 500, abreviação de Standard & Poor's 500, ou simplesmente S&P, 
trata-se de um índice composto por quinhentos ativos cotados nas bolsas de NYSE ou NASDAQ, 
qualificados devido ao seu tamanho de mercado, sua liquidez e sua representação de grupo industrial.
"""

path_dim_tickers = r'G:\Meu Drive\5. Cursos\Pós ML Engineering\Fase 4 - Deep Learning e IA\lstm-stock-predictor-api\dados\aux_data\ativos_ibov.parquet'

# ---------------------------------------------------------------------------------- #
# Funções para gerar as tabelas
# ---------------------------------------------------------------------------------- #

## Tabela dimensão dos tickers

def get_dimension_table_tickers(tickers, path_dim_tickers, save_folder_path):
    dim_tickers = pd.read_parquet(path_dim_tickers, engine = 'pyarrow')
    dim_tickers = dim_tickers[dim_tickers['ticker'].isin(tickers)].sort_values('setor')
    dim_tickers = dim_tickers[['ticker', 'empresa', 'tipo_acao', 'segm_gov', 'setor', 'subsetor', 'segmento']]

    dim_tickers.to_parquet(f"{save_folder_path}/dim_tickers.parquet", engine = 'pyarrow')

    # return dim_tickers

## Histórico de tickers

def get_historical_data_tickers(tickers, save_folder_path, start = '2016-06-20', end = '2026-06-20'):
    df = yf.download(
        [f"{t}.SA" for t in tickers],
        start = start,
        end = end,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    df = df.stack(level="Ticker", future_stack=True).reset_index()
    df["Ticker"] = df["Ticker"].str.replace(".SA", "", regex=False)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['amplitude_pct'] = (df['High'] - df['Low']) / df['Open']
    df['var_dia_pct'] = (df['Close'] - df['Open']) / df['Open']
    df['Date'] = pd.to_datetime(df['Date'])

    df.to_parquet(f"{save_folder_path}/historical_data_tickers.parquet", engine = 'pyarrow')

    # return df

# Histórico dos outros ativos

def get_historical_data_assets(save_folder_path, start = '2016-06-20', end = '2026-06-20'):

    ativos = {
        "^BVSP":"Ibovespa",   # Ibovespa
        "^GSPC":'SP_500',     # S&P 500
        "USDBRL=X":'Dolar'    # Dólar
    }

    df_ativos = yf.download(
        list(ativos.keys()),
        start = start,
        end = end,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    df_ativos = df_ativos.stack(level="Ticker", future_stack=True).reset_index()
    df_ativos["Ticker"] = df_ativos["Ticker"].str.replace(".SA", "", regex=False)
    df_ativos['Date'] = pd.to_datetime(df_ativos['Date']).dt.date
    df_ativos['Ticker'] = df_ativos['Ticker'].map(ativos)

    df_ativos['amplitude_pct'] = (df_ativos['High'] - df_ativos['Low']) / df_ativos['Open']
    df_ativos['var_dia_pct'] = (df_ativos['Close'] - df_ativos['Open']) / df_ativos['Open']

    df_ativos = df_ativos.pivot(index = ['Date'], columns = 'Ticker', values = ['Close', 'amplitude_pct', 'var_dia_pct']).reset_index()
    df_ativos.columns = ['_'.join(map(str, col)).strip('_') for col in df_ativos.columns.values]
    df_ativos['Date'] = pd.to_datetime(df_ativos['Date'])

    df_ativos.to_parquet(f"{save_folder_path}/historical_data_assets.parquet", engine = 'pyarrow')

    # return df_ativos

# Histórico dos índices brasileiros

def get_historical_br_indexes(save_folder_path, start = None):

    if start == None:
        dez_anos_atras = date.today() - relativedelta(years=10)
        dados = sgs.get({"Selic": 11, "IPCA": 433}, start = dez_anos_atras)
    else:
        dados = sgs.get({"Selic": 11, "IPCA": 433}, start = start)

    dados["Selic"] = dados["Selic"].ffill()
    dados["IPCA"] = dados["IPCA"].ffill()
    dados = dados.dropna().reset_index()
    dados['Date'] = pd.to_datetime(dados['Date'])

    dados.to_parquet(f"{save_folder_path}/historical_data_br_indexes.parquet", engine = 'pyarrow')

    # return dados

# ---------------------------------------------------------------------------------- #
# Obtenção da tabela analítica para modelo
# ---------------------------------------------------------------------------------- #



