import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from textblob import TextBlob

# -------------------------------
# CONFIGURATION
# -------------------------------
symbol = "SPY"  # Example: ETF alternative to ES=F for better long-term data
start_date = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
train_size = 0.8
loss_threshold = 0.05  # 5% stop-loss
trail_vol_scale = 0.05  # Trailing stop at 0.05 * volatility
min_confidence_long = 0.6
max_confidence_short = 0.4
capital = 100000  # Starting capital

# -------------------------------
# FETCH DATA
# -------------------------------
data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
data.columns = data.columns.get_level_values(0)
data.dropna(inplace=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# 1. Moving Averages
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# 2. Correct RSI Calculation
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi
data['RSI'] = compute_rsi(data['Close'])

# 3. Volatility (Annualized Standard Deviation of Daily Returns)
data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)

# 4. Volume Change (normalized)
data['VolumePct'] = data['Volume'].pct_change()

# 5. Real Sentiment: MOCK FUNCTION – Replace with real API
def fetch_news_sentiment(date: datetime):
    """
    Replace this with integration to NewsAPI, Alpaca, Twitter, etc.
    For demo: Simulate some variation around the price trend    """
    try:
        # Mock: use sentiment from price change text (placeholder)
        # In production: scrape news and run FinBERT / TextBlob
        from textblob import TextBlob
        sample_texts = [
            f"Stock market up {np.random.uniform(0.5, 2):.1f} percent on optimism",
            f"Concerns over inflation on {date.strftime('%B %d')}",
            f"Economic growth slows, investors cautious on {date.strftime('%Y-%m-%d')}",
            f"Central bank meeting causes volatility in markets"
        ]
        sentiments = [TextBlob(text).sentiment.polarity for text in sample_texts]
        return np.mean(sentiments)
    except:
        return 0.0

data['Sentiment'] = data.index.to_series().apply(fetch_news_sentiment)

# 6. Target: Next day's return direction
data['Return'] = data['Close'].pct_change()
data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)  # 1=up, 0=down

# Drop NaNs
data = data.dropna().copy()

# -------------------------------
# MACHINE LEARNING: XGBoost (Classification)
# -------------------------------
features = ['MA20', 'MA50', 'RSI', 'Volatility', 'VolumePct', 'Sentiment']
target = 'Target'

X = data[features]
y = data[target]

# Time-based split (no shuffle!)
split_idx = int(len(X) * train_size)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train XGBoost
model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_xgb.fit(X_train, y_train)

# Predict probabilities and signals
y_proba = model_xgb.predict_proba(X)[:, 1]
y_pred = model_xgb.predict(X)

data['Predicted'] = pd.Series(y_pred, index=data.index)
data['Confidence'] = pd.Series(y_proba, index=data.index)

# Confidence-based signal generation
data['Signal'] = np.where(
    data['Confidence'] > min_confidence_long, 1,
    np.where(data['Confidence'] < max_confidence_short, -1, 0)
)
data['Signal'] = data['Signal'].shift(1)  # Trade next day

# -------------------------------
# LSTM FOR RETURN PREDICTION (Not used in final signal, for analysis)
# -------------------------------
def create_sequences(X, y, seq_length=10):
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Prepare LSTM data (log returns)
y_lstm = np.log(data['Close'] / data['Close'].shift(1)).dropna().values[1:]
X_lstm_full = data[features].values[len(data)-len(y_lstm):]

X_lstm_train = X_lstm_full[:split_idx + 1]
X_lstm_test = X_lstm_full[split_idx + 1:]
y_lstm_train = y_lstm[:split_idx + 1]
y_lstm_test = y_lstm[split_idx + 1:]

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_lstm_train_scaled = scaler_X.fit_transform(X_lstm_train)
X_lstm_test_scaled = scaler_X.transform(X_lstm_test)
y_lstm_train_scaled = scaler_y.fit_transform(y_lstm_train.reshape(-1, 1))
y_lstm_test_scaled = scaler_y.transform(y_lstm_test.reshape(-1, 1))

# Sequences
seq_length = 10
X_lstm_train_seq, y_lstm_train_seq = create_sequences(X_lstm_train_scaled, y_lstm_train_scaled, seq_length)
X_lstm_test_seq, y_lstm_test_seq = create_sequences(X_lstm_test_scaled, y_lstm_test_scaled, seq_length)

# Build LSTM
model_lstm = Sequential([
    LSTM(50, input_shape=(seq_length, X_lstm_train.shape[1])),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm_train_seq, y_lstm_train_seq, epochs=20, batch_size=32, verbose=0, validation_split=0.1)

# Predict & inverse scale
y_lstm_pred_scaled = model_lstm.predict(X_lstm_test_seq)
y_lstm_pred = scaler_y.inverse_transform(y_lstm_pred_scaled).flatten()
y_lstm_true = y_lstm_test[len(y_lstm_test) - len(y_lstm_pred):]

# -------------------------------
# BACKTEST WITH DYNAMIC SIZING & STOP-LOSS
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

    # Dynamic position sizing (inverse volatility, bounded)
    df['PositionSize'] = 0.05 / (df['Volatility'] + 1e-6)
    df['PositionSize'] = np.clip(df['PositionSize'], 0.01, 0.10)  # 1% to 10%

    # Convert to numpy for faster access
    signals = df['Signal'].to_numpy()
    closes = df['Close'].to_numpy()
    volatility = df['Volatility'].to_numpy()
    pnl = df['PnL'].to_numpy()
    in_trade = df['InTrade'].to_numpy()
    entry_prices = df['EntryPrice'].to_numpy()
    exit_reasons = df['ExitReason'].to_numpy()

    for i in range(1, len(df)):
        if not position and signals[i] == 1 and not np.isnan(closes[i]):  # Buy
            position = 'long'
            entry_price = closes[i]
            peak_price = entry_price
            in_trade[i] = True
            entry_prices[i] = entry_price
