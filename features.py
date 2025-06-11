from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import pandas as pd

def add_indicators(df):
    df = df.copy()

    # Force 'Close' column to a flat 1D array for safety
    close_prices = df['Close'].values.ravel()

    # Re-wrap into a Pandas Series with the same index
    close_series = pd.Series(close_prices, index=df.index)

    # RSI (14-day)
    rsi = RSIIndicator(close=close_series, window=14)
    df['RSI'] = rsi.rsi()

    # MACD (12, 26, 9)
    macd = MACD(close=close_series)
    df['MACD'] = macd.macd_diff()

    # Bollinger Bands (20-day)
    bb = BollingerBands(close=close_series, window=20)
    df['BB_Width'] = bb.bollinger_hband() - bb.bollinger_lband()

    # Drop missing rows (from indicator warm-up)
    df.dropna(inplace=True)
    return df
