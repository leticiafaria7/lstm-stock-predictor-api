
tickers = ['RENT3', 'LREN3', 'SMFT3', 'ABEV3']

path_dim_tickers = r'G:\Meu Drive\5. Cursos\Pós ML Engineering\Fase 4 - Deep Learning e IA\lstm-stock-predictor-api\dados\aux_data\ativos_ibov.parquet'

features = [
    'close', 'high', 'low', 'open', 'volume',
    'close_dolar', 'close_ibovespa', 'close_sp_500', 'selic', 'ipca',
    'ma20', 'ma50', 'bb_upper', 'bb_lower', 'rsi_wilder', 'macd',
    'macd_signal', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos'
]




