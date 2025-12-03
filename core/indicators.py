import pandas as pd
import numpy as np
from datetime import datetime

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def compute_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def compute_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def compute_adx(high, low, close, window=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
    adx = dx.rolling(window).mean()
    return adx, plus_di, minus_di

def fetch_news_sentiment(date: datetime):
    """
    Simulates a sentiment score between -1 and 1.
    Uses a deterministic hash of the date to ensure reproducibility without an external API.
    Simulates a 'monthly regime' + 'daily noise'.
    """
    # Use a fixed seed based on the date integer to be deterministic
    # (year * 10000 + month * 100 + day)
    date_int = date.year * 10000 + date.month * 100 + date.day
    rng = np.random.RandomState(date_int)
    
    # Monthly Trend (Macro sentiment) - varies slowly
    macro_sentiment = np.sin(date.month / 12.0 * 2 * np.pi) * 0.5
    
    # Daily News Shock
    daily_shock = rng.normal(0, 0.3)
    
    sentiment = macro_sentiment + daily_shock
    return np.clip(sentiment, -1.0, 1.0)
