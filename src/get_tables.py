# ---------------------------------------------------------------------------------- #
# Imports
# ---------------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
from bcb.exceptions import SGSError
from datetime import date
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format

# from src.utils import tickers

# ---------------------------------------------------------------------------------- #
# Funções para gerar as tabelas
# ---------------------------------------------------------------------------------- #

## Histórico de tickers

def get_historical_data_tickers(tickers, start = '2016-06-20', end = None):

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    df = yf.download(
        [f"{t}.SA" for t in tickers],
        start = start,
        end = end,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    df = df.stack(level="Ticker", future_stack=True).reset_index()

    if df.empty:
        return pd.DataFrame()
    df["Ticker"] = df["Ticker"].str.replace(".SA", "", regex=False)

    if "Date" not in df.columns:
        return pd.DataFrame()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Date'] = pd.to_datetime(df['Date'])

    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")

    return df

# Histórico dos outros ativos

def get_historical_data_assets(start = '2016-06-20', end = None):

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

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

    df_ativos = df_ativos.pivot(index = ['Date'], columns = 'Ticker', 
                                values = ['Close']).reset_index()
    df_ativos.columns = ['_'.join(map(str, col)).strip('_') for col in df_ativos.columns.values]
    df_ativos['Date'] = pd.to_datetime(df_ativos['Date'])

    return df_ativos

# Histórico dos índices brasileiros

def get_historical_br_indexes(start=None):

    try:
        if start is None:
            start = date.today() - relativedelta(years=10)

        dados = sgs.get({"Selic": 11, "IPCA": 433}, start=start)

    except SGSError:
        return pd.DataFrame(columns=["Date", "Selic", "IPCA"])

    dados["Selic"] = dados["Selic"].ffill()
    dados["IPCA"] = dados["IPCA"].ffill()

    dados = dados.dropna().reset_index()
    dados["Date"] = pd.to_datetime(dados["Date"])

    return dados

# ---------------------------------------------------------------------------------- #
# Funções para cálculos de métricas adicionais
# ---------------------------------------------------------------------------------- #

def calculate_bollinger_bands(df, close_column = 'close'):
    
    rolling = df[close_column].rolling(20)
    std = rolling.std()
    df["bb_middle"] = rolling.mean()
    df["bb_upper"] = df["bb_middle"] + 2 * std
    df["bb_lower"] = df["bb_middle"] - 2 * std

    df = df.drop(columns = 'bb_middle')

    return df

def calculate_rsi(df, close_column = 'close'):
    delta = df[close_column].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # RSI Wilder (média exponencial) --------------------------
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss

    df["rsi_wilder"] = 100 - (100 / (1 + rs))

    return df

def calculate_macd(df, close_column = 'close'):
    ema12 = df[close_column].ewm(span=12, adjust=False).mean()
    ema26 = df[close_column].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df

# ---------------------------------------------------------------------------------- #
# Obtenção da tabela analítica para modelo
# ---------------------------------------------------------------------------------- #

def get_analitical_table(ticker, df_hist_tickers, df_ativos, df_br_indexes):

    # juntar tabelas
    tmp = df_hist_tickers[df_hist_tickers['Ticker'] == ticker].copy()
    tmp = tmp.merge(df_ativos, on = 'Date', how = 'left')
    tmp = tmp.merge(df_br_indexes, on = 'Date', how = 'left')

    tmp.columns = tmp.columns.str.lower()

    # criar coluna de dia da semana e remover finais de semana
    tmp['weekday'] = tmp['date'].dt.day_name()
    tmp = tmp[~tmp['weekday'].isin(['Saturday', 'Sunday'])]

    # filtrar período de dados
    tmp = tmp[tmp['date'] >= pd.to_datetime('2016-07-01')]

    # completar nulls
    tmp['selic'] = tmp['selic'].ffill()
    tmp['ipca'] = tmp['ipca'].ffill()

    for asset in ['sp_500', 'dolar', 'ibovespa']:
        tmp[f"close_{asset}"] = tmp[f"close_{asset}"].ffill()

    # métricas adicionais -------------------------------------------------------------------

    # médias móveis
    tmp["ma20"] = tmp["close"].rolling(20).mean()
    tmp["ma50"] = tmp["close"].rolling(50).mean()

    # bollinger bands (volatilidade)
    # usam média móvel de 20 períodos e ± 2 desvios padrão
    tmp = calculate_bollinger_bands(df = tmp, close_column = 'close')

    # RSI - relative strength index
    # (ação muito comprada ou muito vendida)
    # varia de 0 a 100, < 30 é sobrevendida e > 70 é sobrecomprada
    tmp = calculate_rsi(df = tmp, close_column = 'close')

    # MACD - moving average convergence divergence 
    # (diferença entre duas médias móveis exponenciais)
    # identificar mudanças de tendência, aceleração do movimento e desaceleração
    tmp = calculate_macd(df = tmp, close_column = 'close')

    # criar colunas de codificação cíclica (weekday e mês)
    tmp["weekday_sin"] = np.sin(2 * np.pi * tmp["date"].dt.weekday / 5)
    tmp["weekday_cos"] = np.cos(2 * np.pi * tmp["date"].dt.weekday / 5)

    tmp["month_sin"] = np.sin(2 * np.pi * (tmp["date"].dt.month - 1) / 12)
    tmp["month_cos"] = np.cos(2 * np.pi * (tmp["date"].dt.month - 1) / 12)

    # reordenar colunas
    colunas_idx = ['ticker', 'date', 'weekday']
    tmp = tmp[colunas_idx + [c for c in tmp.columns if c not in colunas_idx]]

    return tmp

