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
from sklearn.preprocessing import StandardScaler

# Bayesian Optimization
try:
    from skopt import gp_minimize, space
    from skopt.utils import use_named_args   
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Bayesian optimization disabled. Install with: pip install scikit-optimize")

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
# Constants
# =============================================================================
BPS_TO_DECIMAL = 10000.0  # Convert basis points to decimal (e.g., 10 bps = 0.001)
ENTRY_EXIT_COST_MULTIPLIER = 2  # Transaction cost applied on both entry and exit

# =============================================================================
# Robust Data Fetching Function with Fallbacks
# =============================================================================
def fetch_stock_data(symbol, start_date, end_date, interval='1d', retries=3):
    """
    Fetch stock data with multiple fallback methods to handle API issues.
    
    Args:
        symbol: Stock symbol (e.g., 'SPY')
        start_date: Start date for data
        end_date: End date for data
        interval: Data interval - '1d', '1h', '15m', '5m' (default: '1d')
        retries: Number of retry attempts (default: 3)
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
                interval=interval, 
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
                interval=interval,
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

def get_bars_per_year(interval):
    """
    Calculate the number of trading bars per year based on the interval.
    
    Args:
        interval: Data interval - '1d', '1h', '15m', '5m'
    
    Returns:
        Number of bars per year for annualization calculations
    """
    # Assumes ~252 trading days per year, 6.5 trading hours per day
    bars_map = {
        '1d': 252,           # Daily bars
        '1h': 252 * 6.5,     # Hourly bars (~1638/year)
        '15m': 252 * 6.5 * 4,  # 15-minute bars (~6552/year)
        '5m': 252 * 6.5 * 12,  # 5-minute bars (~19656/year)
    }
    return bars_map.get(interval, 252)  # Default to daily if unknown

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
        
        html.Label("Data Interval"),
        dcc.Dropdown(
            id='input-interval',
            options=[
                {'label': 'Daily (1 day bars)', 'value': '1d'},
                {'label': 'Hourly (1 hour bars)', 'value': '1h'},
                {'label': '15-Minute bars', 'value': '15m'},
                {'label': '5-Minute bars', 'value': '5m'}
            ],
            value='1d',
            className='input-field',
            clearable=False
        ),
        html.Div("Note: Intraday data limited to last 60-730 days by Yahoo Finance", className='input-hint', style={'fontSize': '11px', 'marginTop': '4px'}),
        
        html.Label("Starting Capital"),
        dcc.Input(id='input-capital', type='number', value=100000, className='input-field'),
        
    html.Label("Long Signal Confidence (%)"),
    dcc.Input(id='slider-min-confidence', type='number', value=75, min=0, max=100, step=0.01, className='input-field'),
    html.Div("Enter percent (0-100). Converted to 0-1 internally.", className='input-hint'),
        
    html.Label("Short Signal Confidence (%)"),
    dcc.Input(id='slider-max-confidence', type='number', value=25, min=0, max=100, step=0.01, className='input-field'),
    html.Div("Enter percent (0-100). Converted to 0-1 internally.", className='input-hint'),
        
    html.Label("Stop-Loss Threshold (%)"),
    dcc.Input(id='slider-loss-threshold', type='number', value=5, min=0.0, max=100.0, step=0.01, className='input-field'),
    html.Div("Enter percent (1-10). Converted to 0-0.1 internally.", className='input-hint'),
        
    html.Label("Trailing Stop Volatility Scale (%)"),
    dcc.Input(id='slider-trail-vol-scale', type='number', value=5, min=0.0, max=100.0, step=0.01, className='input-field'),
    html.Div("Enter percent (0-20). Converted to 0-0.2 internally.", className='input-hint'),
        
    html.Label("XGBoost Trees (n_estimators)"),
    dcc.Input(id='input-n-estimators', type='number', value=500, min=1, max=10000, step=1, className='input-field'),

        html.Label("Train Split (%)"),
        dcc.Input(id='slider-train-pct', type='number', value=65, min=50, max=90, step=1, className='input-field'),
        html.Div("Enter integer percent (50-90). Must be < 100% when combined with validation.", className='input-hint'),

        html.Label("Validation Split (%)"),
        dcc.Input(id='slider-val-pct', type='number', value=15, min=1, max=49, step=1, className='input-field'),
        html.Div("Enter integer percent (1-49). Must be < 100% when combined with training.", className='input-hint'),

        html.Div(id='split-warning', style={'color': 'darkred', 'fontWeight': 'bold', 'marginTop': '6px'}),

        html.Label("Early Stopping Rounds (0 to disable)"),
        dcc.Input(id='input-early-stopping', type='number', value=50, min=0, step=10, className='input-field'),
        
        html.Label("Transaction Cost (bps per trade)"),
        dcc.Input(id='input-transaction-cost', type='number', value=10, min=0, max=100, step=1, className='input-field'),
        html.Div("Basis points (10 bps = 0.10%). Accounts for slippage and commissions.", className='input-hint'),
        
        html.Label("Risk-Free Rate (% annual)"),
        dcc.Input(id='input-risk-free-rate', type='number', value=5.0, min=0, max=15, step=0.1, className='input-field'),
        html.Div("Annual risk-free rate for Sharpe calculation (e.g., T-bill rate).", className='input-hint'),
        
        html.Button('Run Backtest', id='run-button', n_clicks=0, className='run-button'),
        
        html.Hr(),
        html.H5('Bayesian Hyperparameter Optimization'),
        html.Label('Number of Optimization Iterations'),
        dcc.Input(id='bayes-n-calls', type='number', value=10, min=3, max=30, step=1, className='input-field'),
    html.Div('Optimizes (in dashboard order): min_confidence %, max_confidence %, loss_threshold %, trail_vol_scale %, n_estimators, train %, val %, early_stopping', className='input-hint'),
    html.Button('Start Bayesian Optimization', id='bayes-opt-button', n_clicks=0, className='run-button'),
    html.Button('Apply Optimized Params', id='bayes-apply-button', n_clicks=0, className='run-button', disabled=True, style={'marginLeft':'8px'}),
    html.Div(id='bayes-progress', style={'marginTop':'8px', 'fontWeight':'bold'}),
    html.Div(id='bayes-results', style={'marginTop':'12px', 'padding':'8px', 'border':'1px solid #ccc', 'borderRadius':'4px', 'fontSize':'12px'}),
        
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
    dcc.Store(id='bayes-best-params'),
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

# =============================================================================
# Risk Parity Analysis
# =============================================================================
def compute_risk_parity_weights(returns_df):
    """
    Compute risk parity weights for factors/assets.
    Risk parity allocates inversely proportional to volatility.
    """
    try:
        volatilities = returns_df.std()
        if volatilities.sum() == 0:
            return pd.Series(1/len(returns_df), index=returns_df.columns)
        
        # Inverse volatility weighting
        inv_vol = 1.0 / (volatilities + 1e-8)
        weights = inv_vol / inv_vol.sum()
        return weights
    except Exception as e:
        print(f"Error computing risk parity weights: {e}")
        return pd.Series(1/len(returns_df), index=returns_df.columns)

def analyze_risk_parity(data, returns_col='Returns'):
    """
    Analyze portfolio risk contribution by factor.
    Returns risk parity metrics and factor contributions.
    """
    try:
        # Compute daily returns for each factor
        factor_cols = ['MA20', 'MA50', 'RSI', 'Volatility', 'MACD', 'BB_Position', 'ATR', 'ADX']
        available_factors = [col for col in factor_cols if col in data.columns]
        
        if not available_factors:
            return None
        
        # Calculate factor volatility directly (more meaningful than normalized returns)
        factor_vols = {}
        factor_returns_df = pd.DataFrame()
        
        for col in available_factors:
            col_data = data[col].dropna()
            if len(col_data) > 1:
                # Use raw returns/changes
                col_returns = col_data.pct_change().fillna(0)
                factor_returns_df[col] = col_returns
                factor_vols[col] = col_returns.std()
        
        # Compute risk parity weights based on inverse volatility
        vol_array = np.array([factor_vols.get(col, 1e-8) for col in available_factors])
        inv_vol = 1.0 / (vol_array + 1e-8)
        rp_weights = inv_vol / inv_vol.sum()
        rp_weights_series = pd.Series(rp_weights, index=available_factors)
        
        # Compute portfolio returns using risk parity weights
        portfolio_returns = (factor_returns_df * rp_weights).sum(axis=1)
        
        # Risk metrics
        portfolio_vol = portfolio_returns.std()
        cumulative_return = (1 + portfolio_returns).cumprod()
        
        # Factor risk contributions - use volatility as primary metric
        factor_contributions = {col: factor_vols.get(col, 0) for col in available_factors}
        
        return {
            'weights': rp_weights_series,
            'factor_returns': factor_returns_df,
            'portfolio_returns': portfolio_returns,
            'portfolio_vol': portfolio_vol,
            'factor_contributions': factor_contributions,
            'cumulative_return': cumulative_return,
            'factor_vols': factor_vols
        }
    except Exception as e:
        print(f"Error in risk parity analysis: {e}")
        return None

# =============================================================================
# Factor Attribution Analysis
# =============================================================================
def compute_factor_attribution(data, returns_col='Returns'):
    """
    Perform factor attribution analysis to explain returns.
    Identifies which factors contributed most to strategy returns.
    """
    try:
        factor_cols = ['MA20', 'MA50', 'RSI', 'Volatility', 'VolumePct', 'MACD', 
                       'BB_Position', 'ATR', 'ADX', 'Sentiment']
        available_factors = [col for col in factor_cols if col in data.columns]
        
        if not available_factors or returns_col not in data.columns:
            return None
        
        # Prepare data
        X_factors = data[available_factors].fillna(0)
        y_returns = data[returns_col].fillna(0)
        
        # Normalize factors to same scale
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_factors_scaled = scaler.fit_transform(X_factors)
        X_factors_scaled = pd.DataFrame(X_factors_scaled, columns=available_factors, index=data.index)
        
        # Calculate correlation with returns
        correlations = {}
        for col in available_factors:
            corr = X_factors_scaled[col].corr(y_returns)
            correlations[col] = corr
        
        # Calculate impact: rolling window contribution
        window = min(20, len(data) // 4)
        factor_impacts = {}
        
        for col in available_factors:
            rolling_corr = X_factors_scaled[col].rolling(window).corr(y_returns)
            average_impact = rolling_corr.mean()
            max_impact = rolling_corr.max()
            factor_impacts[col] = {
                'avg_impact': average_impact,
                'max_impact': max_impact,
                'correlation': correlations[col]
            }
        
        # Sort by absolute correlation
        sorted_factors = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'factor_correlations': correlations,
            'factor_impacts': factor_impacts,
            'sorted_factors': sorted_factors,
            'top_3_factors': sorted_factors[:3]
        }
    except Exception as e:
        print(f"Error in factor attribution: {e}")
        return None

# =============================================================================
# Bayesian Hyperparameter Optimization
# =============================================================================
def bayesian_optimize_strategy(data, X_train, y_train, X_test, y_test, n_calls=15):
    """
    Use Bayesian optimization to find optimal strategy hyperparameters.
    Optimizes the parameters exposed in the dashboard (keeps the dashboard ordering and formats):
        - min_confidence (percent, integer)
        - max_confidence (percent, integer)
        - loss_threshold (percent, float)
        - trail_vol_scale (percent, float)
        - n_estimators (int)
        - train_pct (percent, int)
        - val_pct (percent, int)
        - early_stopping (int)

    The optimizer works in dashboard order and converts percentages to decimals internally when evaluating.
    """
    if not BAYESIAN_OPT_AVAILABLE:
        print("Bayesian optimization not available. Install scikit-optimize: pip install scikit-optimize")
        return None
    
    try:
        from skopt import gp_minimize, space
        from skopt.utils import use_named_args
        
        # Define search space (use dashboard-friendly ranges / percent units where appropriate)
        search_space = [
            space.Integer(50, 95, name='min_confidence'),       # percent (50-95)
            space.Integer(0, 50, name='max_confidence'),        # percent (0-50)
            space.Real(1.0, 10.0, name='loss_threshold'),       # percent (1-10)
            space.Real(0.0, 20.0, name='trail_vol_scale'),      # percent (0-20)
            space.Integer(50, 2000, name='n_estimators'),       # XGBoost trees
            space.Integer(50, 90, name='train_pct'),            # train percent (50-90)
            space.Integer(1, 49, name='val_pct'),               # val percent (1-49)
            space.Integer(0, 200, name='early_stopping'),       # early stopping rounds (0 disables)
        ]
        # Combine train/test feature frames to allow full-sample prediction when needed
        try:
            X_all = pd.concat([X_train, X_test])
        except Exception:
            # Fallback: if concat fails, attempt to use X_train only
            X_all = X_train.copy()
        
        # Objective function to minimize (negative Sharpe ratio)
        @use_named_args(search_space)
        def objective(**params):
            try:
                # Convert percent inputs from the optimizer into decimals where needed
                min_conf_pct = float(params['min_confidence'])
                max_conf_pct = float(params['max_confidence'])
                loss_thresh_pct = float(params['loss_threshold'])
                trail_vol_pct = float(params['trail_vol_scale'])
                n_est = int(params['n_estimators'])
                train_pct_opt = int(params['train_pct'])
                val_pct_opt = int(params['val_pct'])
                es_opt = int(params['early_stopping'])

                # Validate train/val split: must be < 100
                if train_pct_opt + val_pct_opt >= 100:
                    # Bad configuration -> penalize
                    return 1e6

                # Convert to decimals used by the backtest
                min_conf = min_conf_pct / 100.0
                max_conf = max_conf_pct / 100.0
                loss_thresh = loss_thresh_pct / 100.0
                trail_vol = trail_vol_pct / 100.0

                # Train model
                model = xgb.XGBClassifier(
                    n_estimators=n_est,
                    learning_rate=0.05,
                    max_depth=7,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.5,
                    random_state=42,
                    eval_metric='logloss'
                )
                model.fit(X_train, y_train, verbose=0)

                # Quick local backtest using the same risk-aware simulator
                data_local = data.copy()
                # Use X_all for full-sample probabilities
                data_local['Confidence'] = pd.Series(model.predict_proba(X_all)[:, 1], index=X_all.index)
                data_local['Signal'] = np.where(data_local['Confidence'] > min_conf, 1, np.where(data_local['Confidence'] < max_conf, -1, 0))
                data_local['Signal'] = data_local['Signal'].shift(1)
                data_local = simulate_risk_aware_backtest(data_local, loss_thresh, trail_vol)
                data_local['Returns'] = data_local['PnL'] * data_local['PositionSize']
                strategy_returns = data_local.loc[X_test.index, 'Returns'].values if 'Returns' in data_local.columns else np.zeros(len(X_test))

                # Calculate Sharpe ratio
                if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
                    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
                    # Return negative because gp_minimize minimizes
                    return -sharpe
                else:
                    return 1e3
                    
            except Exception as e:
                print(f"Error in objective function: {e}")
                return 0
        
        # Run Bayesian optimization
        print(f"Starting Bayesian Optimization with {n_calls} iterations...")
        # Allow gp_minimize to run for the requested number of calls
        result = gp_minimize(objective, search_space, n_calls=max(3, int(n_calls)), random_state=42, n_initial_points=min(5, max(1, int(n_calls/3))))
        
        # Extract best parameters
        # Map result vector back to named params (in dashboard order)
        best_params = {
            'min_confidence': int(result.x[0]),
            'max_confidence': int(result.x[1]),
            'loss_threshold': float(result.x[2]),
            'trail_vol_scale': float(result.x[3]),
            'n_estimators': int(result.x[4]),
            'train_pct': int(result.x[5]),
            'val_pct': int(result.x[6]),
            'early_stopping': int(result.x[7]),
            'best_score': -result.fun  # Negate back to get Sharpe ratio
        }
        
        return best_params
        
    except Exception as e:
        print(f"Error in Bayesian optimization: {e}")
        return None

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
     State('input-interval', 'value'),
     State('input-capital', 'value'),
     State('slider-min-confidence', 'value'),
     State('slider-max-confidence', 'value'),
     State('slider-loss-threshold', 'value'),
    State('slider-trail-vol-scale', 'value'),
     State('input-n-estimators', 'value'),
     State('slider-train-pct', 'value'),
     State('slider-val-pct', 'value'),
     State('input-early-stopping', 'value'),
     State('input-transaction-cost', 'value'),
     State('input-risk-free-rate', 'value')]
)
def run_backtest(n_clicks, symbol, start_date, end_date, interval, capital, min_confidence_long, max_confidence_short, loss_threshold_pct, trail_vol_scale, n_estimators, train_pct, val_pct, early_stopping_rounds, transaction_cost_bps, risk_free_rate_pct):
    if n_clicks == 0:
        return html.Div("Set parameters and click 'Run Backtest' to start.", style={'textAlign': 'center', 'marginTop': '50px'}), ''

    try:
        loss_threshold = loss_threshold_pct / 100.0
        trail_vol_scale_val = trail_vol_scale / 100.0
        min_confidence = min_confidence_long / 100.0
        max_confidence = max_confidence_short / 100.0
        
        # Convert transaction cost from basis points to decimal (e.g., 10 bps = 0.001)
        transaction_cost = (transaction_cost_bps if transaction_cost_bps else 10) / BPS_TO_DECIMAL
        
        # Convert risk-free rate from percentage to decimal (e.g., 5% = 0.05)
        risk_free_rate = (risk_free_rate_pct if risk_free_rate_pct else 5.0) / 100.0
        
        # Get bars per year for annualization based on interval
        bars_per_year = get_bars_per_year(interval if interval else '1d')
        
        # 1. Fetch & Feature Engineering
        data = fetch_stock_data(symbol if symbol else 'SPY', start_date, end_date, interval if interval else '1d')
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
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(bars_per_year)
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

        # IMPORTANT: Walk-forward validation for continuous signals across entire period
        # Train on expanding window, predict on next batch (realistic out-of-sample predictions)
        
        y_proba_all = np.zeros(len(X))
        walk_forward_window = max(100, train_size // 2)  # Train on expanding window, step by 10%
        step_size = max(1, len(X) // 20)  # Step forward by ~5% of data at a time
        
        print(f"Walk-forward validation: window={walk_forward_window}, step={step_size}, total_samples={len(X)}")
        
        # Initial training period (no signals yet - bootstrapping)
        y_proba_all[:walk_forward_window] = np.nan
        
        # Walk-forward loop
        for start_idx in range(walk_forward_window, len(X) - step_size, step_size):
            end_idx = min(start_idx + step_size, len(X))
            
            # Train on data up to start_idx
            X_wf_train = X[:start_idx]
            y_wf_train = y[:start_idx]
            
            # Predict on next batch
            X_wf_pred = X[start_idx:end_idx]
            
            try:
                # Train model on accumulated data
                wf_model = xgb.XGBClassifier(
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
                wf_model.fit(X_wf_train, y_wf_train, verbose=0)
                
                # Generate out-of-sample predictions
                y_proba_all[start_idx:end_idx] = wf_model.predict_proba(X_wf_pred)[:, 1]
            except Exception as e:
                print(f"Walk-forward error at idx {start_idx}: {e}")
                y_proba_all[start_idx:end_idx] = np.nan
        
        # Handle any remaining data at the end
        if end_idx < len(X):
            try:
                wf_model = xgb.XGBClassifier(
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
                wf_model.fit(X[:end_idx], y[:end_idx], verbose=0)
                X_wf_final = X[end_idx:]
                y_proba_all[end_idx:] = wf_model.predict_proba(X_wf_final)[:, 1]
            except Exception as e:
                print(f"Final walk-forward error: {e}")
        
        data['Confidence'] = pd.Series(y_proba_all, index=data.index)
        data['Signal'] = np.where(data['Confidence'] > min_confidence, 1, 
                                  np.where(data['Confidence'] < max_confidence, -1, 0))
        # Don't generate signals during initial bootstrap period (where confidence is NaN)
        data.loc[data['Confidence'].isna(), 'Signal'] = 0
        data['Signal'] = data['Signal'].shift(1)

        # 3. Backtest with transaction costs
        data = simulate_risk_aware_backtest(data, loss_threshold, trail_vol_scale_val)
        
        # Calculate returns with transaction costs
        # Apply transaction cost on each trade entry and exit
        data['TradeEntry'] = (data['Signal'].diff().abs() > 0).astype(int)
        data['TransactionCosts'] = data['TradeEntry'] * transaction_cost * ENTRY_EXIT_COST_MULTIPLIER
        data['Returns'] = data['PnL'] * data['PositionSize'] - data['TransactionCosts']
        data['Cumulative Returns'] = (1 + data['Returns']).cumprod() * capital

        # 4. Performance Metrics with proper Sharpe calculation
        final_value = data['Cumulative Returns'].iloc[-1] if not data['Cumulative Returns'].empty else capital
        total_return = (final_value / capital) - 1
        annualized_return = (1 + total_return) ** (bars_per_year / len(data)) - 1 if len(data) > 0 else 0
        daily_returns = data['Returns'].fillna(0)
        std_daily = daily_returns.std()
        
        # Calculate excess return and proper Sharpe ratio
        # Sharpe = (Annualized Return - Risk Free Rate) / Annualized Volatility
        volatility_annual = std_daily * np.sqrt(bars_per_year)
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility_annual if volatility_annual > 1e-6 else 0
        
        peak = data['Cumulative Returns'].cummax()
        drawdown = peak - data['Cumulative Returns']
        max_drawdown_val = drawdown.max()
        max_drawdown_pct = (max_drawdown_val / peak.max()) if peak.max() > 0 else 0

        volatility_annual = std_daily * np.sqrt(bars_per_year)
        win_rate = (data['Returns'] > 0).sum() / (data['PnL'] != 0).sum() if (data['PnL'] != 0).sum() > 0 else 0
        
        # Calculate total transaction costs paid
        total_txn_costs = data['TransactionCosts'].sum() * capital
        
        summary_text = f"""
============================================================
📊 STRATEGY PERFORMANCE SUMMARY
============================================================
Symbol:                    {symbol.upper()}
Interval:                  {interval if interval else '1d'}
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
  (Risk-Free Rate:         {risk_free_rate:.2%})
  (Excess Return:          {excess_return:.2%})
Max Drawdown:              {max_drawdown_pct:.2%}
Win Rate (Trades):         {win_rate:.2%}
============================================================
Trade Count:               {(data['PnL'] != 0).sum()}
Transaction Costs Paid:    ${total_txn_costs:,.2f}
Avg Position Size:         {data[data['PnL'] != 0]['PositionSize'].mean():.2%}
============================================================
⚠️  NOTE: Walk-forward validation used
    Continuously trained on expanding historical window
    Ensures all predictions are truly out-of-sample
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

        # Compute Risk Parity and Factor Attribution Analysis
        rp_analysis = analyze_risk_parity(data)
        factor_attr = compute_factor_attribution(data)
        
        # Risk Parity Visualization
        rp_viz_div = html.Div()
        if rp_analysis:
            rp_weights = rp_analysis['weights']
            
            # Sort weights for better visualization
            rp_weights_sorted = rp_weights.sort_values(ascending=False)
            fig_rp = go.Figure(data=[go.Bar(
                x=rp_weights_sorted.index, 
                y=rp_weights_sorted.values,
                marker_color='steelblue',
                text=[f'{v:.4f}' for v in rp_weights_sorted.values],
                textposition='auto'
            )])
            fig_rp.update_layout(
                title='Risk Parity Factor Weights (Inverse Volatility)',
                xaxis_title='Factor',
                yaxis_title='Weight',
                height=400
            )
            
            # Factor volatility pie chart
            factor_vols = rp_analysis.get('factor_vols', {})
            if factor_vols:
                # Filter out very small volatilities
                sig_vols = {k: v for k, v in factor_vols.items() if v > 1e-8}
                if sig_vols:
                    fig_contrib = go.Figure(data=[go.Pie(
                        labels=list(sig_vols.keys()),
                        values=list(sig_vols.values()),
                        marker=dict(colors=px.colors.sequential.Viridis),
                        textposition='inside',
                        textinfo='label+percent'
                    )])
                    fig_contrib.update_layout(
                        title='Factor Volatility Contribution to Risk',
                        height=400
                    )
                else:
                    # All volatilities too small - show equal distribution
                    fig_contrib = go.Figure(data=[go.Pie(
                        labels=list(factor_vols.keys()),
                        values=[1]*len(factor_vols),
                        marker=dict(colors=px.colors.sequential.Viridis),
                        textposition='inside',
                        textinfo='label+percent'
                    )])
                    fig_contrib.update_layout(
                        title='Equal Risk Distribution',
                        height=400
                    )
            else:
                fig_contrib = go.Figure().add_annotation(text="No volatility data available")
            
            rp_viz_div = html.Div([
                dcc.Graph(figure=fig_rp),
                dcc.Graph(figure=fig_contrib)
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'})
        
        # Factor Attribution Visualization
        factor_attr_div = html.Div()
        if factor_attr:
            correlations = factor_attr['factor_correlations']
            sorted_factors = factor_attr['sorted_factors']
            
            # Bar chart of factor correlations
            fig_corr = go.Figure(data=[go.Bar(
                x=[f[0] for f in sorted_factors],
                y=[f[1] for f in sorted_factors],
                marker_color=['green' if f[1] > 0 else 'red' for f in sorted_factors]
            )])
            fig_corr.update_layout(title='Factor Correlation with Strategy Returns', xaxis_title='Factor', yaxis_title='Correlation')
            
            # Top 3 factors text
            top_3_text = "Top Factors Contributing to Returns:\n"
            for i, (factor, corr) in enumerate(factor_attr['top_3_factors'][:3], 1):
                impact_info = factor_attr['factor_impacts'].get(factor, {})
                top_3_text += f"\n{i}. {factor}: {corr:.3f} correlation"
                if 'avg_impact' in impact_info:
                    top_3_text += f" (avg impact: {impact_info['avg_impact']:.4f})"
            
            factor_attr_div = html.Div([
                dcc.Graph(figure=fig_corr),
                html.Pre(top_3_text, style={'backgroundColor': '#f0f0f0', 'padding': '12px', 'borderRadius': '4px', 'fontSize': '12px'})
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
        
        # Add Risk Parity Analysis Section
        if rp_analysis:
            results_sections.append(
                html.Div(className='results-section', children=[
                    html.Div(className='section-header', children=[
                        html.H3('Risk Parity Analysis')
                    ]),
                    rp_viz_div
                ])
            )
        
        # Add Factor Attribution Section
        if factor_attr:
            results_sections.append(
                html.Div(className='results-section', children=[
                    html.Div(className='section-header', children=[
                        html.H3('Factor Attribution Analysis')
                    ]),
                    factor_attr_div
                ])
            )
        
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
    [State('exp-n-estimators', 'value'), State('exp-train-pct', 'value'), State('exp-val-pct', 'value'), State('exp-early-stopping', 'value'), State('exp-mode', 'value'), State('exp-n-trials', 'value'), State('input-symbol','value'), State('input-start-date','value'), State('input-end-date','value'), State('input-interval','value'), State('slider-min-confidence','value'), State('slider-max-confidence','value'), State('slider-loss-threshold','value'), State('slider-trail-vol-scale','value')]
)
def run_experiments(n_clicks, n_est_str, train_str, val_str, es_str, mode, n_trials, symbol, start_date, end_date, interval, min_conf, max_conf, loss_threshold_pct, trail_vol_scale):
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
    thread = threading.Thread(target=_run_experiments_bg, args=(n_est_str, train_str, val_str, es_str, mode, n_trials, symbol, start_date, end_date, interval, min_conf, max_conf, loss_threshold_pct, trail_vol_scale))
    thread.daemon = True
    thread.start()
    
    return 'Experiments running in background... Check progress below.'

def _run_experiments_bg(n_est_str, train_str, val_str, es_str, mode, n_trials, symbol, start_date, end_date, interval, min_conf, max_conf, loss_threshold_pct, trail_vol_scale):
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
        bars_per_year = get_bars_per_year(interval if interval else '1d')
        data = fetch_stock_data(symbol if symbol else 'SPY', start_date, end_date, interval if interval else '1d')
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
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(bars_per_year)
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

            # quick backtest using signals - ONLY on out-of-sample data to avoid look-ahead bias
            data_local = data.copy()
            
            # Only use out-of-sample predictions
            y_proba_all = np.zeros(len(X_all))
            y_proba_all[:train_size] = np.nan  # No trading on training data
            y_proba_all[train_size:train_size+test_size] = model.predict_proba(X_test)[:, 1]
            # Validation data (if any)
            if len(X_all) > train_size + test_size:
                X_val_local = X_all[train_size+test_size:]
                y_proba_all[train_size+test_size:] = model.predict_proba(X_val_local)[:, 1]
            
            data_local['Confidence'] = pd.Series(y_proba_all, index=X_all.index)
            data_local['Signal'] = np.where(data_local['Confidence'] > min_conf, 1, np.where(data_local['Confidence'] < max_conf, -1, 0))
            data_local.loc[data_local['Confidence'].isna(), 'Signal'] = 0
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


@app.callback(
    Output('bayes-apply-button', 'disabled'),
    Input('bayes-best-params', 'data')
)
def enable_apply_button(best_params):
    # Enable the Apply button when optimizer results are available
    return False if best_params else True


@app.callback(
    [Output('slider-min-confidence', 'value'), Output('slider-max-confidence', 'value'), Output('slider-loss-threshold', 'value'), Output('slider-trail-vol-scale', 'value'), Output('input-n-estimators', 'value'), Output('slider-train-pct', 'value'), Output('slider-val-pct', 'value'), Output('input-early-stopping', 'value')],
    Input('bayes-apply-button', 'n_clicks'),
    State('bayes-best-params', 'data'),
    prevent_initial_call=True
)
def apply_bayes_params(n_clicks, best_params):
    # When user clicks Apply, write the stored optimizer values into the UI inputs
    if not best_params:
        # nothing to apply
        return [dash.no_update] * 8

    try:
        # Quantize and coerce values to types the UI and validation expect:
        # - confidences: round to 2 decimals (UI accepts percent floats)
        # - thresholds/scales: round to 2 decimals
        # - n_estimators, train/val pct, early_stopping: integers
        min_conf = best_params.get('min_confidence', None)
        max_conf = best_params.get('max_confidence', None)
        loss_thr = best_params.get('loss_threshold', None)
        trail_vol = best_params.get('trail_vol_scale', None)
        n_est = best_params.get('n_estimators', None)
        train_pct = best_params.get('train_pct', None)
        val_pct = best_params.get('val_pct', None)
        es = best_params.get('early_stopping', None)

        # Rounding / coercion rules
        if min_conf is not None:
            min_conf = round(float(min_conf), 2)
        if max_conf is not None:
            max_conf = round(float(max_conf), 2)
        if loss_thr is not None:
            loss_thr = round(float(loss_thr), 2)
        if trail_vol is not None:
            trail_vol = round(float(trail_vol), 2)
        if n_est is not None:
            try:
                n_est = int(round(float(n_est)))
            except Exception:
                n_est = int(n_est)
        if train_pct is not None:
            train_pct = int(round(float(train_pct)))
        if val_pct is not None:
            val_pct = int(round(float(val_pct)))
        if es is not None:
            es = int(round(float(es)))

        # Ensure train + val < 100 (adjust val_pct if needed)
        if train_pct is not None and val_pct is not None:
            if train_pct + val_pct >= 100:
                # reduce val_pct to keep sum < 100, keep at least 1
                val_pct = max(1, 99 - train_pct)

        return [
            min_conf if min_conf is not None else dash.no_update,
            max_conf if max_conf is not None else dash.no_update,
            loss_thr if loss_thr is not None else dash.no_update,
            trail_vol if trail_vol is not None else dash.no_update,
            n_est if n_est is not None else dash.no_update,
            train_pct if train_pct is not None else dash.no_update,
            val_pct if val_pct is not None else dash.no_update,
            es if es is not None else dash.no_update
        ]
    except Exception:
        return [dash.no_update] * 8


@app.callback(
    [Output('bayes-progress', 'children'), Output('bayes-results', 'children'), Output('bayes-best-params', 'data')],
    Input('bayes-opt-button', 'n_clicks'),
    [State('input-symbol', 'value'),
     State('input-start-date', 'value'),
     State('input-end-date', 'value'),
     State('input-interval', 'value'),
     State('slider-train-pct', 'value'),
     State('slider-val-pct', 'value'),
     State('bayes-n-calls', 'value')]
)
def run_bayesian_optimization(n_clicks, symbol, start_date, end_date, interval, train_pct, val_pct, n_calls):
    if not n_clicks or not BAYESIAN_OPT_AVAILABLE:
        return '', '', None
    
    try:
        # Fetch data
        bars_per_year = get_bars_per_year(interval if interval else '1d')
        data = fetch_stock_data(symbol if symbol else 'SPY', start_date, end_date, interval if interval else '1d')
        if data is None or data.empty:
            return 'Error fetching data', ''
        
        # Feature engineering (simplified)
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = compute_rsi(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(bars_per_year)
        data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = compute_macd(data['Close'])
        data['Return'] = data['Close'].pct_change()
        data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)
        
        data = data.dropna().copy()
        
        if len(data) < 100:
            return 'Not enough data', ''
        
        # Prepare train/test split
        features = ['MA20', 'MA50', 'RSI', 'Volatility', 'MACD']
        X, y = data[features], data['Target']
        
        train_pct_val = int(train_pct) if train_pct else 65
        val_pct_val = int(val_pct) if val_pct else 15
        
        train_size = int(len(X) * (train_pct_val / 100.0))
        test_size = int(len(X) * ((100 - train_pct_val - val_pct_val) / 100.0))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:train_size+test_size], y[train_size:train_size+test_size]
        
        # Run Bayesian optimization
        best_params = bayesian_optimize_strategy(data, X_train, y_train, X_test, y_test, n_calls=n_calls)

        if best_params:
            # Format results in the same style/order as the dashboard (percent values where the UI displays %)
            results_text = f"""
✓ Bayesian Optimization Complete!

Best Parameters Found (dashboard order):
  • Min Confidence: {best_params['min_confidence']} %
  • Max Confidence: {best_params['max_confidence']} %
  • Loss Threshold: {best_params['loss_threshold']:.2f} %
  • Trailing Vol Scale: {best_params['trail_vol_scale']:.2f} %
  • XGBoost Trees: {int(best_params['n_estimators'])}
  • Train Split: {best_params['train_pct']} %
  • Validation Split: {best_params['val_pct']} %
  • Early Stopping Rounds: {best_params['early_stopping']}
  
Expected Sharpe Ratio: {best_params['best_score']:.3f}

Tip: Copy these values into the main backtest controls (they are formatted the same way the dashboard expects).
            """
            # return results plus the params (store them so the Apply button can use them)
            return '✓ Optimization Complete', results_text, best_params
        else:
            return '✗ Optimization Failed', 'Check console for errors. Is scikit-optimize installed?', None
            
    except Exception as e:
        print(f"Error in Bayesian optimization: {e}")
        return f'✗ Error: {str(e)}', '', None


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8050)
