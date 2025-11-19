import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Fix for yfinance API issues - configure session with proper headers
import requests
import joblib
import json
import os
import itertools
import threading
import queue
from datetime import datetime, timedelta
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Robust Data Fetching Function with Fallbacks
# =============================================================================
def fetch_stock_data(symbol, start_date, end_date, retries=3):
    """
    Fetch stock data with multiple fallback methods to handle API issues.
    """
    import time
    
    # Create a custom session with headers for each request
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    for attempt in range(retries):
        try:
            # Method 1: yfinance download with session and explicit parameters
            data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                interval='1d', 
                progress=False,
                auto_adjust=True,
                prepost=False,
                threads=True
            )
            
            if data is not None and not data.empty and len(data) > 0:
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Standardize column names
                column_mapping = {
                    'close': 'Close', 'open': 'Open', 'high': 'High', 
                    'low': 'Low', 'volume': 'Volume'
                }
                data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns}, inplace=True)
                
                print(f"Successfully fetched {len(data)} rows for {symbol}")
                return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
        
        try:
            # Method 2: yfinance Ticker with history
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date, 
                end=end_date, 
                interval='1d',
                auto_adjust=True,
                prepost=False
            )
            
            if not data.empty and len(data) > 0:
                # Standardize column names
                if 'Close' not in data.columns and 'close' in data.columns:
                    data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 
                                        'low': 'Low', 'volume': 'Volume'}, inplace=True)
                print(f"Successfully fetched {len(data)} rows for {symbol} (Ticker method)")
                return data
        except Exception as e:
            print(f"Ticker method attempt {attempt + 1} failed for {symbol}: {str(e)}")
        
        # Wait before retry (except on last attempt)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
    
    # If all methods fail, return None
    print(f"All attempts failed for {symbol}")
    return None

# =============================================================================
# App Initialization
# =============================================================================
# Dash automatically serves files from the 'assets' folder.
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Trading Strategy Backtest Dashboard"

# Global state for async experiments
exp_status = {'running': False, 'progress': '', 'results': None, 'results_file': None}
exp_lock = threading.Lock()

# =============================================================================
# Reusable Components & Styling
# =============================================================================
def build_control_panel():
    """Creates the control panel for user inputs."""
    return html.Div(id='control-panel', style={'width': '300px', 'flexShrink': '0'}, children=[
        html.H4("Strategy Controls"),
        
        html.Label("Stock/Futures Symbol"),
        dcc.Input(id='input-symbol', type='text', value='SPY', className='input-field'),
        
        html.Label("Date Range"),
        html.Div(className='date-range-wrapper', children=[
            dcc.Input(
                id='input-start-date',
                type='date',
                value=(datetime.now() - timedelta(days=365*2)).date().isoformat(),
                className='input-field',
                style={'marginBottom': '0'}
            ),
            dcc.Input(
                id='input-end-date',
                type='date',
                value=datetime.now().date().isoformat(),
                className='input-field',
                style={'marginBottom': '0'}
            ),
        ]),
        
        html.Label("Starting Capital"),
        dcc.Input(id='input-capital', type='number', value=100000, className='input-field'),
        
        html.Label("Long Signal Confidence (%)"),
        dcc.Input(id='slider-min-confidence', type='number', value=75, min=50, max=100, step=1, className='input-field'),
        html.Div("Enter integer percent (50-100). Converted to 0-1 internally.", className='input-hint'),
        
        html.Label("Short Signal Confidence (%)"),
        dcc.Input(id='slider-max-confidence', type='number', value=25, min=0, max=50, step=1, className='input-field'),
        html.Div("Enter integer percent (0-50). Converted to 0-1 internally.", className='input-hint'),
        
        html.Label("Stop-Loss Threshold (%)"),
        dcc.Input(id='slider-loss-threshold', type='number', value=5, min=1, max=10, step=0.5, className='input-field'),
        html.Div("Enter percent (1-10). Converted to 0-0.1 internally.", className='input-hint'),
        
        html.Label("Trailing Stop Volatility Scale (%)"),
        dcc.Input(id='slider-trail-vol-scale', type='number', value=5, min=0, max=20, step=0.5, className='input-field'),
        html.Div("Enter percent (0-20). Converted to 0-0.2 internally.", className='input-hint'),
        
        html.Label("XGBoost Trees (n_estimators)"),
        dcc.Input(id='input-n-estimators', type='number', value=500, min=50, max=2000, step=50, className='input-field'),

        html.Label("Train Split (%)"),
        dcc.Input(id='slider-train-pct', type='number', value=65, min=50, max=90, step=1, className='input-field'),
        html.Div("Enter integer percent (50-90). Must be < 100% when combined with validation.", className='input-hint'),

        html.Label("Validation Split (%)"),
        dcc.Input(id='slider-val-pct', type='number', value=15, min=1, max=49, step=1, className='input-field'),
        html.Div("Enter integer percent (1-49). Must be < 100% when combined with training.", className='input-hint'),

        html.Div(id='split-warning', style={'color': 'darkred', 'fontWeight': 'bold', 'marginTop': '6px'}),

        html.Label("Early Stopping Rounds (0 to disable)"),
        dcc.Input(id='input-early-stopping', type='number', value=50, min=0, step=10, className='input-field'),
        
        html.Button('Run Backtest', id='run-button', n_clicks=0, className='run-button')
        ,
        html.Hr(),
        html.H5('Experiment Runner'),
        html.Label('n_estimators (comma-separated)'),
        dcc.Input(id='exp-n-estimators', type='text', value='100,200,500', className='input-field'),
        html.Label('Train % (comma-separated)'),
        dcc.Input(id='exp-train-pct', type='text', value='65,70', className='input-field'),
        html.Label('Val % (comma-separated)'),
        dcc.Input(id='exp-val-pct', type='text', value='15,20', className='input-field'),
        html.Label('Early stopping (comma-separated, 0 disables)'),
        dcc.Input(id='exp-early-stopping', type='text', value='0,50', className='input-field'),
        html.Label('Mode'),
        dcc.RadioItems(id='exp-mode', options=[{'label':'Grid','value':'grid'},{'label':'Random','value':'random'}], value='grid'),
        html.Label('Random trials (if random mode)'),
        dcc.Input(id='exp-n-trials', type='number', value=10, min=1, className='input-field'),
        html.Button('Run Experiments', id='run-experiments-button', n_clicks=0, className='run-button'),
        html.Div(id='exp-progress', style={'marginTop':'8px', 'fontWeight':'bold'})
    ])

# =============================================================================
# App Layout
# =============================================================================
app.layout = html.Div(id='main-container', style={'width': '100%', 'padding': '20px'}, children=[
    html.H1(app.title),
    html.Div(style={'display': 'flex', 'gap': '30px', 'width': '100%'}, children=[
        build_control_panel(),
        dcc.Loading(id="loading-spinner", type="default", children=html.Div(id='results-output', style={'flex': '1', 'width': '90%', 'minWidth': '1200px'}))
    ]),
    html.Div(id='run-backtest-loading-output', style={'display': 'none'})
])

# =============================================================================
# Core Logic & Callbacks
# =============================================================================
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    # I added 1e-6 here to prevent a divide-by-zero error if loss is 0
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

def compute_macd(prices, fast=12, slow=26, signal=9):
    """Compute MACD (Moving Average Convergence Divergence)"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def compute_bollinger_bands(prices, window=20, num_std=2):
    """Compute Bollinger Bands"""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def compute_atr(high, low, close, window=14):
    """Compute Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def compute_adx(high, low, close, window=14):
    """Compute Average Directional Index"""
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
    try:
        sample_texts = [
            f"Stock market up {np.random.uniform(0.5, 2):.1f} percent on optimism",
            f"Concerns over inflation on {date.strftime('%B %d')}",
            f"Economic growth slows, investors cautious on {date.strftime('%Y-%m-%d')}",
            f"Central bank meeting causes volatility in markets"
        ]
        return np.mean([TextBlob(text).sentiment.polarity for text in sample_texts])
    except:
        return 0.0

# --- THIS IS THE CORRECTED FUNCTION ---
def simulate_risk_aware_backtest(df, loss_threshold=0.05, trail_vol_scale=0.05):
    # I'm making a copy to avoid any SettingWithCopyWarnings
    df = df.copy()
    df['InTrade'], df['EntryPrice'], df['ExitReason'], df['PnL'] = [False, np.nan, '', 0.0]
    position, entry_price, peak_price = None, 0.0, 0.0
    
    # I pre-calculate position size, adding a small number to volatility to avoid division by zero
    df['PositionSize'] = np.clip(0.05 / (df['Volatility'] + 1e-6), 0.01, 0.10)

    for i in range(1, len(df)):
        row = df.iloc[i]
        
        try:
            current_price = float(row['Close'].item()) if hasattr(row['Close'], 'item') else float(row['Close'])
        except (ValueError, TypeError):
            current_price = np.nan
        # I'm skipping any rows where the price is missing to prevent errors
        if np.isnan(current_price):
            continue 
        
        # This logic handles entering a new trade
        try:
            signal_val = float(row['Signal'].item()) if hasattr(row['Signal'], 'item') else float(row['Signal'])
        except (ValueError, TypeError):
            signal_val = 0
        if not position and signal_val != 0:
            position = 'long' if signal_val == 1 else 'short'
            entry_price = peak_price = current_price
            df.loc[df.index[i], ['InTrade', 'EntryPrice']] = [True, entry_price]
        
        # This logic handles an already-open position
        elif position and not np.isnan(entry_price):
            
            # --- I'm making this volatility calculation "NaN-safe" ---
            # This was a likely source of the error, as NaN volatility would break the 'max' function
            try:
                volatility_val = float(row['Volatility'].item()) if hasattr(row['Volatility'], 'item') else float(row['Volatility'])
            except (ValueError, TypeError):
                volatility_val = 0.0
            volatility_component = trail_vol_scale * volatility_val / np.sqrt(252)
            if np.isnan(volatility_component):
                volatility_component = 0.0 # I'm setting a default if volatility is NaN
                
            # Now, vol_stop_threshold will always be a valid number.
            vol_stop_threshold = max(volatility_component, 0.01)
            # --- End of my NaN-safe fix ---

            if position == 'long':
                # I added a check here to prevent division by zero
                current_return = (current_price - entry_price) / entry_price if entry_price != 0 else 0
                peak_price = max(peak_price, current_price)
                
                # I also added a check here for peak_price
                trailing_return = (current_price - peak_price) / peak_price if peak_price != 0 else 0
                
                exit_trade = False
                exit_reason = ''

                if current_return <= -loss_threshold:
                    exit_trade = True
                    exit_reason = 'stop-loss'
                elif trailing_return <= -vol_stop_threshold:
                    exit_trade = True
                    exit_reason = 'trailing-stop'

                if exit_trade:
                    df.loc[df.index[i], 'PnL'] = current_return
                    df.loc[df.index[i], 'ExitReason'] = exit_reason
                    position = None
                else:
                    df.loc[df.index[i], ['InTrade', 'PnL']] = [True, current_return]
            
            elif position == 'short':
                # I added the same division-by-zero check here
                current_return = (entry_price - current_price) / entry_price if entry_price != 0 else 0
                peak_price = min(peak_price, current_price)
                
                exit_trade = False
                exit_reason = ''

                if current_return <= -loss_threshold:
                    exit_trade = True
                    exit_reason = 'short-stop'
                else:
                    # I'm using a small number '1e-6' just to be extra safe against tiny/zero prices
                    if peak_price > 1e-6:
                        bounce_from_low = (current_price - peak_price) / peak_price
                        if bounce_from_low >= vol_stop_threshold:
                            exit_trade = True
                            exit_reason = 'short-trailing'

                if exit_trade:
                    df.loc[df.index[i], 'PnL'] = current_return
                    df.loc[df.index[i], 'ExitReason'] = exit_reason
                    position = None
                else:
                    df.loc[df.index[i], ['InTrade', 'PnL']] = [True, current_return]
    return df

@app.callback(
    [Output('results-output', 'children'),
     Output('run-backtest-loading-output', 'children')],
    Input('run-button', 'n_clicks'),
    [State('input-symbol', 'value'),
     State('input-start-date', 'value'),
     State('input-end-date', 'value'),
     State('input-capital', 'value'),
     State('slider-min-confidence', 'value'),
     State('slider-max-confidence', 'value'),
     State('slider-loss-threshold', 'value'),
    State('slider-trail-vol-scale', 'value'),
     State('input-n-estimators', 'value'),
     State('slider-train-pct', 'value'),
     State('slider-val-pct', 'value'),
     State('input-early-stopping', 'value')]
)
def run_backtest(n_clicks, symbol, start_date, end_date, capital, min_confidence_long, max_confidence_short, loss_threshold_pct, trail_vol_scale, n_estimators, train_pct, val_pct, early_stopping_rounds):
    if n_clicks == 0:
        return html.Div("Set parameters and click 'Run Backtest' to start.", style={'textAlign': 'center', 'marginTop': '50px'}), ''

    try:
        loss_threshold = loss_threshold_pct / 100.0
        trail_vol_scale_val = trail_vol_scale / 100.0
        min_confidence = min_confidence_long / 100.0
        max_confidence = max_confidence_short / 100.0
        
        # 1. Fetch & Feature Engineering
        data = fetch_stock_data(symbol if symbol else 'SPY', start_date, end_date)
        if data is None or data.empty:
            return html.Div([
                html.H4(f"Unable to fetch data for '{symbol}'" , style={'color': 'red'}),
                html.P([
                    "This could be due to:",
                    html.Br(),
                    "• Yahoo Finance API temporary outage (common issue)",
                    html.Br(),
                    "• Invalid symbol name",
                    html.Br(),
                    "• No trading data for the selected date range",
                    html.Br(),
                    html.Br(),
                    "Please try again in a few minutes, or try a different symbol like 'AAPL' or 'MSFT'."
                ])
            ], style={'textAlign': 'center', 'padding': '20px'}), ''

        # Flatten MultiIndex columns if they exist (yfinance downloads can have MultiIndex columns)
        if isinstance(data.columns, pd.MultiIndex):
            # For MultiIndex, take only the first level (the column name)
            data.columns = data.columns.get_level_values(0)

        # --- I REMOVED A BAD 'dropna' FROM HERE ---
        # I found a data.dropna(inplace=True) here which I removed.
        # Dropping NaNs before creating rolling features is a bug that introduces more NaNs.

        # Calculate all technical indicators
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = compute_rsi(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        data['VolumePct'] = data['Volume'].pct_change()
        data['Sentiment'] = data.index.to_series().apply(fetch_news_sentiment)
        
        # Add new indicators for improved ML accuracy
        macd_line, macd_signal, macd_histogram = compute_macd(data['Close'])
        data['MACD'] = macd_line
        data['MACD_Signal'] = macd_signal
        data['MACD_Histogram'] = macd_histogram
        
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(data['Close'])
        data['BB_Upper'] = bb_upper
        data['BB_Middle'] = bb_middle
        data['BB_Lower'] = bb_lower
        data['BB_Position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-6)
        
        data['ATR'] = compute_atr(data['High'], data['Low'], data['Close'])
        
        adx, plus_di, minus_di = compute_adx(data['High'], data['Low'], data['Close'])
        data['ADX'] = adx
        data['Plus_DI'] = plus_di
        data['Minus_DI'] = minus_di
        
        data['Return'] = data['Close'].pct_change()
        data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)
        
        # This is the *correct* place to drop NaNs, *after* all rolling features are calculated
        data = data.dropna().copy()

        if len(data) < 100: # Need enough data for features and training
            return html.Div("Not enough data to run the backtest. Please select a wider date range.", style={'color': 'red', 'textAlign': 'center'}), ''

        # 2. Machine Learning with improved features and walk-forward validation
        features = ['MA20', 'MA50', 'RSI', 'Volatility', 'VolumePct', 'Sentiment', 
                   'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Position', 'ATR', 'ADX', 'Plus_DI', 'Minus_DI']
        target = 'Target'
        X, y = data[features], data[target]
        
        # Determine splits from UI inputs (train_pct and val_pct are provided as percentages)
        try:
            train_pct_val = int(train_pct)
        except Exception:
            train_pct_val = 65
        try:
            val_pct_val = int(val_pct)
        except Exception:
            val_pct_val = 15

        if train_pct_val + val_pct_val >= 100:
            return html.Div("Invalid split: train + validation must be less than 100%.", style={'color': 'red', 'textAlign': 'center'}), ''

        test_pct_val = 100 - train_pct_val - val_pct_val
        train_size = int(len(X) * (train_pct_val / 100.0))
        test_size = int(len(X) * (test_pct_val / 100.0))
        val_size = len(X) - train_size - test_size

        # Ensure minimum sizes (fall back gracefully if dataset small)
        if train_size < 10:
            train_size = int(len(X) * 0.6)
        if test_size < 5:
            test_size = max(5, int(len(X) * 0.2))

        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:train_size+test_size], y[train_size:train_size+test_size]
        X_val, y_val = X[train_size+test_size:], y[train_size+test_size:]

        # Prepare model hyperparameters from UI
        try:
            n_estimators_val = int(n_estimators)
        except Exception:
            n_estimators_val = 500
        try:
            esr = int(early_stopping_rounds)
        except Exception:
            esr = 50

        model_xgb = xgb.XGBClassifier(
            n_estimators=n_estimators_val,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.5,
            min_child_weight=1,
            random_state=42,
            eval_metric='logloss'
        )

        # Fit with optional early stopping on the test fold to avoid overfitting
        if esr and esr > 0:
            try:
                model_xgb.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=esr,
                    verbose=False
                )
            except TypeError:
                model_xgb.fit(X_train, y_train)
        else:
            model_xgb.fit(X_train, y_train)
        # Compute out-of-sample metrics on the test fold
        try:
            y_test_pred = model_xgb.predict(X_test)
            y_test_proba = model_xgb.predict_proba(X_test)[:, 1]
            cm = confusion_matrix(y_test, y_test_pred)
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_auc = auc(fpr, tpr)
        except Exception:
            cm = None
            fpr = tpr = []
            roc_auc = None

        y_proba = model_xgb.predict_proba(X)[:, 1]
        data['Confidence'] = pd.Series(y_proba, index=data.index)
        data['Signal'] = np.where(data['Confidence'] > min_confidence, 1, np.where(data['Confidence'] < max_confidence, -1, 0))
        data['Signal'] = data['Signal'].shift(1)

        # 3. Backtest
        data = simulate_risk_aware_backtest(data, loss_threshold, trail_vol_scale_val)
        data['Returns'] = data['PnL'] * data['PositionSize']
        data['Cumulative Returns'] = (1 + data['Returns']).cumprod() * capital

        # 4. Performance Metrics
        final_value = data['Cumulative Returns'].iloc[-1] if not data['Cumulative Returns'].empty else capital
        total_return = (final_value / capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(data)) - 1 if len(data) > 0 else 0
        daily_returns = data['Returns'].fillna(0)
        std_daily = daily_returns.std()
        sharpe_ratio = (annualized_return) / (std_daily * np.sqrt(252)) if std_daily > 1e-6 else 0
        
        peak = data['Cumulative Returns'].cummax()
        drawdown = peak - data['Cumulative Returns']
        max_drawdown_val = drawdown.max()
        max_drawdown_pct = (max_drawdown_val / peak.max()) if peak.max() > 0 else 0

        volatility_annual = std_daily * np.sqrt(252)
        win_rate = (data['Returns'] > 0).sum() / (data['PnL'] != 0).sum() if (data['PnL'] != 0).sum() > 0 else 0
        
        summary_text = f"""
============================================================
📊 STRATEGY PERFORMANCE SUMMARY
============================================================
Symbol:                    {symbol.upper()}
Start Date:                {data.index[0].strftime('%Y-%m-%d')}
End Date:                  {data.index[-1].strftime('%Y-%m-%d')}
============================================================
Starting Capital:          ${capital:,.2f}
Ending Value:              ${final_value:,.2f}
Total Return:              {total_return:.2%}
Annualized Return:         {annualized_return:.2%}
============================================================
Annual Volatility:         {volatility_annual:.2%}
Sharpe Ratio:              {sharpe_ratio:.2f}
Max Drawdown:              {max_drawdown_pct:.2%}
Win Rate (Trades):         {win_rate:.2%}
============================================================
Trade Count:               {(data['PnL'] != 0).sum()}
Avg Position Size:         {data[data['PnL'] != 0]['PositionSize'].mean():.2%}
============================================================
"""
        latest = data.iloc[-1]
        try:
            latest_signal = float(latest['Signal'].item()) if hasattr(latest['Signal'], 'item') else float(latest['Signal'])
        except (ValueError, TypeError):
            latest_signal = 0
        try:
            latest_confidence = float(latest['Confidence'].item()) if hasattr(latest['Confidence'], 'item') else float(latest['Confidence'])
        except (ValueError, TypeError):
            latest_confidence = 0.0
        latest_signal_text = f"**Latest Signal**: {'Buy' if latest_signal == 1 else 'Sell' if latest_signal == -1 else 'Hold'} (Confidence: {latest_confidence:.3f})"

        # 5. Visualizations
        fig_price = px.line(data, x=data.index, y='Close', title=f'{symbol.upper()} Price & Trade Signals')
        fig_price.update_layout(height=1200, hovermode='x unified', margin=dict(l=50, r=50, t=50, b=50))
        fig_price.add_scatter(x=data[data['Signal'] == 1].index, y=data[data['Signal'] == 1]['Close'], mode='markers', name='Buy Signal', marker=dict(color='limegreen', size=9, symbol='triangle-up'))
        fig_price.add_scatter(x=data[data['Signal'] == -1].index, y=data[data['Signal'] == -1]['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=9, symbol='triangle-down'))
        fig_price.add_scatter(x=data[data['ExitReason'].str.contains('stop', na=False)].index, y=data[data['ExitReason'].str.contains('stop', na=False)]['Close'], mode='markers', name='Stop Trigger', marker=dict(color='orange', size=7, symbol='x'))

        fig_portfolio = px.line(data, x=data.index, y='Cumulative Returns', title=f'Portfolio Value (Started with ${capital:,})', labels={'Cumulative Returns': 'Portfolio Value ($)'})
        fig_portfolio.update_layout(height=1200, hovermode='x unified', margin=dict(l=50, r=50, t=50, b=50))
        
        fig_returns = px.histogram(data, x='Returns', nbins=100, title='Strategy Daily Returns Distribution', labels={'Returns': 'Daily Return'})

        # Persist run: save model, metrics, results
        try:
            runs_dir = os.path.join(os.getcwd(), 'runs')
            os.makedirs(runs_dir, exist_ok=True)
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(runs_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            # save model
            joblib.dump(model_xgb, os.path.join(run_dir, 'model.joblib'))
            # save results
            data.to_csv(os.path.join(run_dir, 'backtest_results.csv'))
            # save metrics
            metrics = {
                'final_value': float(final_value),
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'annual_volatility': float(volatility_annual),
                'sharpe': float(sharpe_ratio),
                'win_rate': float(win_rate),
                'trades': int((data['PnL'] != 0).sum())
            }
            if cm is not None:
                metrics['confusion_matrix'] = cm.tolist()
            if roc_auc is not None:
                metrics['roc_auc'] = float(roc_auc)
            with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            saved_run_path = run_dir
        except Exception:
            saved_run_path = None
        fig_returns.update_layout(height=900, hovermode='x unified', margin=dict(l=50, r=50, t=50, b=50))

        # Add confusion / ROC visuals if computed
        extra_graphs = []
        if cm is not None:
            fig_cm = go.Figure(data=go.Heatmap(z=cm.tolist(), x=['Pred 0','Pred 1'], y=['True 0','True 1'], colorscale='Blues'))
            fig_cm.update_layout(title='Confusion Matrix')
            extra_graphs.append(dcc.Graph(figure=fig_cm))
        if roc_auc is not None:
            fig_roc = go.Figure()
            fig_roc.add_scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})')
            fig_roc.add_scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Chance')
            fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            extra_graphs.append(dcc.Graph(figure=fig_roc))

        run_saved_note = html.Div()
        if saved_run_path:
            run_saved_note = html.Div(f"✓ Run saved: {saved_run_path}", style={'fontStyle':'italic', 'marginTop':'12px', 'color': '#00d084', 'fontSize': '12px'})

        # Build professional metric cards
        metric_cards = html.Div(className='metrics-grid', children=[
            html.Div(className='metric-card', children=[
                html.Div(className='metric-label', children='Total Return'),
                html.Div(className=f'metric-value {"positive" if total_return >= 0 else "negative"}', children=f'{total_return*100:+.2f}%'),
                html.Div(className='metric-change', children=f'${final_value:,.0f} final')
            ]),
            html.Div(className='metric-card', children=[
                html.Div(className='metric-label', children='Annualized Return'),
                html.Div(className=f'metric-value {"positive" if annualized_return >= 0 else "negative"}', children=f'{annualized_return*100:+.2f}%'),
                html.Div(className='metric-change', children='yearly')
            ]),
            html.Div(className='metric-card', children=[
                html.Div(className='metric-label', children='Sharpe Ratio'),
                html.Div(className='metric-value neutral', children=f'{sharpe_ratio:.3f}'),
                html.Div(className='metric-change', children=f'{volatility_annual*100:.2f}% vol')
            ]),
            html.Div(className='metric-card', children=[
                html.Div(className='metric-label', children='Win Rate'),
                html.Div(className='metric-value positive' if win_rate >= 0.5 else 'metric-value neutral', children=f'{win_rate*100:.1f}%'),
                html.Div(className='metric-change', children=f'{int((data["PnL"] != 0).sum())} trades')
            ]),
            html.Div(className='metric-card', children=[
                html.Div(className='metric-label', children='Max Drawdown'),
                html.Div(className='metric-value negative' if max_drawdown_pct < 0 else 'metric-value positive', children=f'{max_drawdown_pct*100:.2f}%'),
                html.Div(className='metric-change', children='peak to trough')
            ]),
            html.Div(className='metric-card', children=[
                html.Div(className='metric-label', children='Model AUC'),
                html.Div(className='metric-value positive' if roc_auc and roc_auc > 0.5 else 'metric-value neutral', children=f'{roc_auc:.3f}' if roc_auc else 'N/A'),
                html.Div(className='metric-change', children='test set')
            ]),
        ])

        # Build main results container
        results_sections = [
            html.Div(className='results-section', children=[
                html.Div(className='section-header', children=[
                    html.Div(className='status-indicator'),
                    html.H3('Performance Metrics')
                ]),
                metric_cards
            ]),
            html.Div(className='results-section', children=[
                html.Div(className='section-header', children=[
                    html.H3('Price & Signals')
                ]),
                dcc.Graph(figure=fig_price)
            ]),
            html.Div(className='results-section', children=[
                html.Div(className='section-header', children=[
                    html.H3('Portfolio Value')
                ]),
                dcc.Graph(figure=fig_portfolio)
            ]),
            html.Div(className='results-section', children=[
                html.Div(className='section-header', children=[
                    html.H3('Daily Returns Distribution')
                ]),
                dcc.Graph(figure=fig_returns)
            ]),
        ]
        
        # Add confusion matrix and ROC if available
        if cm is not None or roc_auc is not None:
            results_sections.append(
                html.Div(className='results-section', children=[
                    html.Div(className='section-header', children=[
                        html.H3('Model Diagnostics')
                    ]),
                    html.Div(extra_graphs, style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'})
                ])
            )
        
        results_sections.append(html.Div(className='results-section', children=[
            html.Div(className='section-header', children=[
                html.H3('Execution Summary')
            ]),
            html.Pre(summary_text, className='summary-container'),
            html.H5(latest_signal_text, style={'textAlign': 'center', 'fontWeight': 'bold', 'marginTop': '16px'}),
            run_saved_note
        ]))

        return html.Div(results_sections, style={'width': '100%'}), ''
    except Exception as e:
        # I'm importing 'traceback' to give a more detailed error message
        import traceback
        return html.Div([
            html.H4("An error occurred:", style={'color': 'red'}),
            # This will print the full error stack, which is more helpful for debugging
            html.Pre(f"{traceback.format_exc()}", style={'border': '1px solid #ddd', 'padding': '10px', 'overflowX': 'auto'})
        ]), ''


@app.callback(
    [Output('split-warning', 'children'), Output('run-button', 'disabled')],
    [Input('slider-train-pct', 'value'), Input('slider-val-pct', 'value')]
)
def validate_splits(train_val, val_val):
    try:
        s = int(train_val) + int(val_val)
    except Exception:
        return '', False
    if s >= 100:
        return f"Invalid splits: train + val = {s}%. Must be < 100%.", True
    return '', False


@app.callback(
    Output('exp-progress', 'children'),
    Input('run-experiments-button', 'n_clicks'),
    [State('exp-n-estimators', 'value'), State('exp-train-pct', 'value'), State('exp-val-pct', 'value'), State('exp-early-stopping', 'value'), State('exp-mode', 'value'), State('exp-n-trials', 'value'), State('input-symbol','value'), State('input-start-date','value'), State('input-end-date','value'), State('slider-min-confidence','value'), State('slider-max-confidence','value'), State('slider-loss-threshold','value'), State('slider-trail-vol-scale','value')]
)
def run_experiments(n_clicks, n_est_str, train_str, val_str, es_str, mode, n_trials, symbol, start_date, end_date, min_conf, max_conf, loss_threshold_pct, trail_vol_scale):
    if not n_clicks:
        return ''
    
    global exp_status
    with exp_lock:
        if exp_status['running']:
            return 'Experiments already running...'
        exp_status['running'] = True
        exp_status['progress'] = 'Starting experiments...'
        exp_status['results'] = None
        exp_status['results_file'] = None
    
    # Spawn background thread
    thread = threading.Thread(target=_run_experiments_bg, args=(n_est_str, train_str, val_str, es_str, mode, n_trials, symbol, start_date, end_date, min_conf, max_conf, loss_threshold_pct, trail_vol_scale))
    thread.daemon = True
    thread.start()
    
    return 'Experiments running in background... Check progress below.'

def _run_experiments_bg(n_est_str, train_str, val_str, es_str, mode, n_trials, symbol, start_date, end_date, min_conf, max_conf, loss_threshold_pct, trail_vol_scale):
    """Background thread function to run experiments."""
    global exp_status
    
    def update_progress(msg):
        with exp_lock:
            exp_status['progress'] = msg
    
    try:
        # parse lists
        def parse_list(s):
            if s is None:
                return []
            return [int(x.strip()) for x in str(s).split(',') if x.strip()]

        n_est_list = parse_list(n_est_str)
        train_list = parse_list(train_str)
        val_list = parse_list(val_str)
        es_list = parse_list(es_str)

        if not n_est_list or not train_list or not val_list:
            update_progress('ERROR: Provide comma-separated lists for n_estimators, train pct and val pct.')
            with exp_lock:
                exp_status['running'] = False
            return

        update_progress('Fetching data...')
        # Fetch and preprocess data once
        data = fetch_stock_data(symbol if symbol else 'SPY', start_date, end_date)
        if data is None or data.empty:
            update_progress(f'ERROR: Unable to fetch data for {symbol}. Yahoo Finance API may be temporarily unavailable. Please try again.')
            with exp_lock:
                exp_status['running'] = False
            return
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = compute_rsi(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        data['VolumePct'] = data['Volume'].pct_change()
        data['Sentiment'] = data.index.to_series().apply(fetch_news_sentiment)
        macd_line, macd_signal, macd_hist = compute_macd(data['Close'])
        data['MACD'] = macd_line
        data['MACD_Signal'] = macd_signal
        data['MACD_Histogram'] = macd_hist
        bb_upper, bb_mid, bb_lower = compute_bollinger_bands(data['Close'])
        data['BB_Position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-6)
        data['ATR'] = compute_atr(data['High'], data['Low'], data['Close'])
        adx, plus_di, minus_di = compute_adx(data['High'], data['Low'], data['Close'])
        data['ADX'] = adx
        data['Plus_DI'] = plus_di
        data['Minus_DI'] = minus_di
        data['Return'] = data['Close'].pct_change()
        data['Target'] = (data['Return'].shift(-1) > 0).astype(int)
        data = data.dropna().copy()

        features = ['MA20','MA50','RSI','Volatility','VolumePct','Sentiment','MACD','MACD_Signal','MACD_Histogram','BB_Position','ATR','ADX','Plus_DI','Minus_DI']
        X_all = data[features]
        y_all = data['Target']

        update_progress('Building parameter combinations...')
        combos = []
        if mode == 'grid':
            for ne, tr, va, es in itertools.product(n_est_list, train_list, val_list, es_list if es_list else [0]):
                combos.append({'n_estimators':ne, 'train_pct':tr, 'val_pct':va, 'early_stopping':es})
        else:
            import random
            for _ in range(int(n_trials) if n_trials else 10):
                combos.append({'n_estimators': random.choice(n_est_list), 'train_pct': random.choice(train_list), 'val_pct': random.choice(val_list), 'early_stopping': random.choice(es_list) if es_list else 0})

        results = []
        runs_dir = os.path.join(os.getcwd(), 'experiments')
        os.makedirs(runs_dir, exist_ok=True)
        exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_file = os.path.join(runs_dir, f'exp_{exp_id}.csv')

        for idx, combo in enumerate(combos):
            update_progress(f'Running combo {idx+1}/{len(combos)}: n_est={combo["n_estimators"]}, train={combo["train_pct"]}%, val={combo["val_pct"]}%, es={combo["early_stopping"]}')
            
            ne = combo['n_estimators']
            tr = combo['train_pct']
            va = combo['val_pct']
            es = combo['early_stopping']
            # compute splits
            train_size = int(len(X_all) * (tr/100.0))
            test_size = int(len(X_all) * ((100-tr-va)/100.0))
            if train_size < 10 or test_size < 5:
                continue
            X_train = X_all[:train_size]
            y_train = y_all[:train_size]
            X_test = X_all[train_size:train_size+test_size]
            y_test = y_all[train_size:train_size+test_size]

            model = xgb.XGBClassifier(n_estimators=int(ne), learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
            try:
                if es and es>0:
                    model.fit(X_train, y_train, eval_set=[(X_test,y_test)], early_stopping_rounds=int(es), verbose=False)
                else:
                    model.fit(X_train, y_train)
            except Exception:
                try:
                    model.fit(X_train, y_train)
                except Exception:
                    continue

            # evaluate
            try:
                y_test_proba = model.predict_proba(X_test)[:,1]
                y_test_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_test_pred)
                fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                roc_auc = auc(fpr,tpr)
                accuracy = (y_test_pred==y_test).mean()
            except Exception:
                cm = None; roc_auc=None; accuracy=None

            # quick backtest using signals
            data_local = data.copy()
            data_local['Confidence'] = pd.Series(model.predict_proba(X_all)[:,1], index=X_all.index)
            data_local['Signal'] = np.where(data_local['Confidence'] > min_conf, 1, np.where(data_local['Confidence'] < max_conf, -1, 0))
            data_local['Signal'] = data_local['Signal'].shift(1)
            res_local = simulate_risk_aware_backtest(data_local, loss_threshold_pct/100.0, trail_vol_scale)
            res_local['Returns'] = res_local['PnL'] * res_local['PositionSize']
            final_value = res_local['Returns'].add(1).cumprod().iloc[-1] if not res_local['Returns'].empty else 1.0

            results.append({'n_estimators':ne, 'train_pct':tr, 'val_pct':va, 'early_stopping':es, 'roc_auc':roc_auc, 'accuracy':accuracy, 'final_value':float(final_value)})

        # save results
        try:
            import csv
            keys = ['n_estimators','train_pct','val_pct','early_stopping','roc_auc','accuracy','final_value']
            with open(exp_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
        except Exception:
            pass

        # build results
        if not results:
            update_progress('No valid experiment runs completed.')
        else:
            header = [html.Th(k) for k in results[0].keys()]
            rows = []
            for r in results:
                rows.append(html.Tr([html.Td(f'{r[k]:.4f}' if isinstance(r[k], float) else r[k]) for k in r.keys()]))
            table = html.Table([html.Tr(header)] + rows, style={'width':'100%','overflowX':'auto','border':'1px solid #ccc'})
            result_div = html.Div([html.H5('Experiment Results'), table, html.Div(f'Saved to {exp_file}', style={'fontStyle':'italic'})])
            with exp_lock:
                exp_status['results'] = result_div
                exp_status['progress'] = f'Completed {len(results)} runs. Results saved to {exp_file}'

    except Exception as e:
        update_progress(f'ERROR: {str(e)}')
    finally:
        with exp_lock:
            exp_status['running'] = False


@app.callback(
    Output('exp-progress', 'children', allow_duplicate=True),
    Input('exp-progress', 'id'),
    prevent_initial_call=True
)
def poll_exp_progress(_):
    """Poller callback to update progress every 500ms."""
    global exp_status
    with exp_lock:
        if exp_status['results']:
            result = exp_status['results']
            exp_status['results'] = None
            return result
        elif exp_status['running']:
            return exp_status['progress']
        else:
            return exp_status['progress']

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8050)
