import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split
import ta

# -------------------------------
# FETCH DATA
# -------------------------------
symbol = "SPY"
start_date = "2020-01-01"
end_date = "2023-01-01"
data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
data.columns = data.columns.get_level_values(0)
data.dropna(inplace=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
features = []
indicator_bb = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
data['bb_bbm'] = indicator_bb.bollinger_mavg()
data['bb_bbh'] = indicator_bb.bollinger_hband()
data['bb_bbl'] = indicator_bb.bollinger_lband()
features.extend(['bb_bbm', 'bb_bbh', 'bb_bbl'])

indicator_macd = ta.trend.MACD(close=data["Close"], window_slow=26, window_fast=12, window_sign=9)
data['macd'] = indicator_macd.macd()
data['macd_signal'] = indicator_macd.macd_signal()
features.extend(['macd', 'macd_signal'])

indicator_ichimoku = ta.trend.IchimokuIndicator(high=data["High"], low=data["Low"], window1=9, window2=26, window3=52)
data['ichimoku_a'] = indicator_ichimoku.ichimoku_a()
data['ichimoku_b'] = indicator_ichimoku.ichimoku_b()
features.extend(['ichimoku_a', 'ichimoku_b'])

data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
features.extend(['MA20', 'MA50'])

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi
data['RSI'] = compute_rsi(data['Close'])
features.append('RSI')

data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
data['VolumePct'] = data['Volume'].pct_change()
features.extend(['Volatility', 'VolumePct'])

data['Return'] = data['Close'].pct_change()
data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)
data = data.dropna().copy()

# -------------------------------
# MACHINE LEARNING: XGBoost (Classification)
# -------------------------------
target = 'Target'
X = data[features]
y = data[target]

train_size = 0.8
split_idx = int(len(X) * train_size)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_xgb.fit(X_train, y_train)

y_proba = model_xgb.predict_proba(X)[:, 1]
y_pred = model_xgb.predict(X)

data['Predicted'] = pd.Series(y_pred, index=data.index)
data['Confidence'] = pd.Series(y_proba, index=data.index)

min_confidence_long = 0.6
max_confidence_short = 0.4
data['Signal'] = np.where(
    data['Confidence'] > min_confidence_long, 1,
    np.where(data['Confidence'] < max_confidence_short, -1, 0)
)
data['Signal'] = data['Signal'].shift(1)

# -------------------------------
# BACKTEST
# -------------------------------
def simulate_risk_aware_backtest(df, loss_threshold=0.05, trail_vol_scale=0.05):
    df = df.copy()
    df['InTrade'] = False
    df['EntryPrice'] = np.nan
    df['ExitReason'] = ''
    df['PnL'] = 0.0
    position = None
    entry_price = 0.0
    peak_price = 0.0

    df['PositionSize'] = 0.05 / (df['Volatility'] + 1e-6)
    df['PositionSize'] = np.clip(df['PositionSize'], 0.01, 0.10)

    signals = df['Signal'].to_numpy()
    closes = df['Close'].to_numpy()
    volatility = df['Volatility'].to_numpy()
    pnl = df['PnL'].to_numpy()
    in_trade = df['InTrade'].to_numpy()
    entry_prices = df['EntryPrice'].to_numpy()
    exit_reasons = df['ExitReason'].to_numpy()

    for i in range(1, len(df)):
        if not position and signals[i] == 1 and not np.isnan(closes[i]):
            position = 'long'
            entry_price = closes[i]
            peak_price = entry_price
            in_trade[i] = True
            entry_prices[i] = entry_price
        elif not position and signals[i] == -1 and not np.isnan(closes[i]):
            position = 'short'
            entry_price = closes[i]
            peak_price = entry_price
            in_trade[i] = True
            entry_prices[i] = entry_price
        elif position == 'long' and not np.isnan(entry_price):
            current_price = closes[i]
            current_return = (current_price - entry_price) / entry_price
            peak_price = max(peak_price, current_price)
            trailing_return = (current_price - peak_price) / peak_price
            vol_stop_threshold = max(trail_vol_scale * volatility[i] / np.sqrt(252), 0.01)

            if current_return <= -loss_threshold:
                pnl[i] = current_return
                exit_reasons[i] = 'stop-loss'
                position = None
            elif trailing_return <= -vol_stop_threshold:
                pnl[i] = current_return
                exit_reasons[i] = 'trailing-stop'
                position = None
            else:
                in_trade[i] = True
                pnl[i] = current_return
        elif position == 'short' and not np.isnan(entry_price):
            current_price = closes[i]
            current_return = (entry_price - current_price) / entry_price
            peak_price = min(peak_price, current_price)
            trailing_return = (peak_price - current_price) / peak_price
            vol_stop_threshold = max(trail_vol_scale * volatility[i] / np.sqrt(252), 0.01)

            if current_return <= -loss_threshold:
                pnl[i] = current_return
                exit_reasons[i] = 'short-stop'
                position = None
            elif trailing_return >= vol_stop_threshold:
                pnl[i] = current_return
                exit_reasons[i] = 'short-trailing'
                position = None
            else:
                in_trade[i] = True
                pnl[i] = current_return
        else:
            pnl[i] = 0.0

    df['PnL'] = pnl
    df['InTrade'] = in_trade
    df['EntryPrice'] = entry_prices
    df['ExitReason'] = exit_reasons
    return df

data = data.loc[~data.index.duplicated(keep='first')]
data = simulate_risk_aware_backtest(data, 0.05, 0.05)

print(data.tail())
