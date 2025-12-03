import dash
from dash import dcc, html, Input, Output, State, no_update, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Core Imports
from core.data import fetch_stock_data, get_bars_per_year
from core.strategy import Strategy
from core.indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_adx,
    fetch_news_sentiment,
)
from core.backtest import simulate_risk_aware_backtest
from core.utils import (
    analyze_risk_parity,
    compute_factor_attribution,
    compute_cross_validation,
    calculate_advanced_metrics,
    calculate_trade_metrics
)

# Bayesian Optimization
try:
    from skopt import gp_minimize, space
    from skopt.utils import use_named_args
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Bayesian optimization disabled. Install with: pip install scikit-optimize")

import joblib
import os
import itertools
import threading
import time
from datetime import datetime, timedelta
import warnings
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Constants
# =============================================================================
BPS_TO_DECIMAL = 10000.0  # Convert basis points to decimal (e.g., 10 bps = 0.001)
ENTRY_EXIT_COST_MULTIPLIER = 2  # Transaction cost applied on both entry and exit

# =============================================================================
# App Initialization
# =============================================================================
# Dash automatically serves files from the 'assets' folder.
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Trading Strategy Backtest Dashboard"

# Global state for async experiments
exp_status = {
    'running': False, 
    'progress': '', 
    'results': None, 
    'results_file': None,
    'start_time': None,
    'total_trials': 0,
    'completed_trials': 0
}
exp_lock = threading.Lock()

# Global state for async backtest
backtest_status = {'running': False, 'progress': 'Ready', 'result': None}
backtest_lock = threading.Lock()

# =============================================================================
# Reusable Components & Styling
# =============================================================================
def build_control_panel():
    """Creates the control panel for user inputs."""
    return html.Div(id='control-panel', children=[
        html.H4("Strategy Controls"),
        
        html.Label("Stock/Futures Symbol"),
        dcc.Input(id='input-symbol', type='text', value='SPY', className='input-field'),
        
        html.Label("Date Range"),
        html.Div(className='date-range-wrapper', children=[
            dcc.Input(
                id='input-start-date',
                type='text',
                value=(datetime.now() - timedelta(days=365*2)).date().isoformat(),
                className='input-field',
                style={'marginBottom': '0'},
                placeholder='YYYY-MM-DD'
            ),
            dcc.Input(
                id='input-end-date',
                type='text',
                value=datetime.now().date().isoformat(),
                className='input-field',
                style={'marginBottom': '0'},
                placeholder='YYYY-MM-DD'
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
        html.Div("Note: Intraday data limited to last 60-730 days by Yahoo Finance", className='input-hint'),
        
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

        html.Div(id='split-warning', style={'color': '#de350b', 'fontWeight': '600', 'fontSize': '12px', 'marginTop': '6px'}),

        html.Label("Early Stopping Rounds"),
        dcc.Input(id='input-early-stopping', type='number', value=50, min=0, step=10, className='input-field'),
        
        html.Label("Transaction Cost (bps per trade)"),
        dcc.Input(id='input-transaction-cost', type='number', value=10, min=0, max=100, step=1, className='input-field'),
        html.Div("Basis points (10 bps = 0.10%). Accounts for slippage and commissions.", className='input-hint'),
        
        html.Label("Risk-Free Rate (% annual)"),
        dcc.Input(id='input-risk-free-rate', type='number', value=5.0, min=0, max=15, step=0.1, className='input-field'),
        html.Div("Annual risk-free rate for Sharpe calculation (e.g., T-bill rate).", className='input-hint'),
        
        html.Button('Run Backtest', id='run-button', n_clicks=0, className='run-button'),
        
        html.Hr(),
        html.H5("Comparison Tools"),
        html.Button('Set Current as Baseline', id='btn-lock-baseline', n_clicks=0, className='run-button', style={'backgroundColor': '#42526e', 'marginTop': '0'}, disabled=True),
        html.Div(id='baseline-status', style={'fontSize': '11px', 'color': '#006644', 'marginTop': '8px', 'fontWeight': '600'}),

        html.Hr(),
        html.H5('Bayesian Hyperparameter Optimization'),
        html.Label('Number of Optimization Iterations'),
        dcc.Input(id='bayes-n-calls', type='number', value=10, min=3, max=30, step=1, className='input-field'),
        html.Div('Optimizes params in order. See code for details.', className='input-hint'),
        html.Button('Start Bayesian Optimization', id='bayes-opt-button', n_clicks=0, className='run-button'),
        html.Button('Apply Optimized Params', id='bayes-apply-button', n_clicks=0, className='run-button', disabled=True, style={'marginTop':'8px'})
        ,html.Div(id='bayes-progress', style={'marginTop':'8px', 'fontWeight':'600', 'fontSize':'12px'}),
        html.Div(id='bayes-results', style={'marginTop':'12px', 'padding':'8px', 'border':'1px solid #ebecf0', 'borderRadius':'3px', 'fontSize':'12px'}),
            
        html.Hr(),
        html.H5('Experiment Runner'),
        html.Label('n_estimators (comma-separated)'),
        dcc.Input(id='exp-n-estimators', type='text', value='100,200,500', className='input-field'),
        html.Label('Train % (comma-separated)'),
        dcc.Input(id='exp-train-pct', type='text', value='65,70', className='input-field'),
        html.Label('Val % (comma-separated)'),
        dcc.Input(id='exp-val-pct', type='text', value='15,20', className='input-field'),
        html.Label('Early stopping (comma-separated)'),
        dcc.Input(id='exp-early-stopping', type='text', value='0,50', className='input-field'),
        html.Label('Mode'),
        dcc.RadioItems(id='exp-mode', options=[{'label':' Grid Search ','value':'grid'},{'label':' Random Search ','value':'random'}], value='grid', labelStyle={'display': 'inline-block', 'marginRight': '10px', 'fontSize': '12px'})
        ,html.Label('Random trials'),
        dcc.Input(id='exp-n-trials', type='number', value=10, min=1, className='input-field'),
        html.Button('Run Experiments', id='run-experiments-button', n_clicks=0, className='run-button'),
        html.Div(id='exp-progress', style={'marginTop':'8px', 'fontWeight':'600', 'fontSize':'12px'})
    ])

# =============================================================================
# App Layout
# =============================================================================
app.layout = html.Div(id='main-container', children=[
    dcc.Store(id='bayes-best-params'),
    dcc.Store(id='current-run-store'),
    dcc.Store(id='baseline-run-store'),
    dcc.Interval(id='backtest-interval', interval=500, n_intervals=0, disabled=True),
    dcc.Interval(id='experiment-interval', interval=1000, n_intervals=0, disabled=True),
    html.H1(app.title),
    html.Div(className='app-layout', children=[
        build_control_panel(),
        html.Div(children=[
            html.Div(id='comparison-output'),
            html.Div(id='results-output', children=[
                html.Div(className='results-section', style={'textAlign': 'center', 'color': '#505f79'}, children=[
                    html.H3("Ready to Backtest", style={'color': '#172b4d'}),
                    html.P("Configure parameters on the left and click 'Run Backtest' to begin analysis.")
                ])
            ])
        ])
    ]),
    html.Div(id='run-backtest-loading-output', style={'display': 'none'})
])

# =============================================================================
# Core Logic & Callbacks
# =============================================================================

# Note: Core logic functions have been moved to the 'core' package.
# - Indicators: core/indicators.py
# - Data: core/data.py
# - Backtesting: core/backtest.py
# - Utils: core/utils.py

# =============================================================================
# Background Execution Logic
# =============================================================================

def bayesian_optimize_strategy(data, X_train, y_train, X_test, y_test, n_calls=15):
    if not BAYESIAN_OPT_AVAILABLE:
        return None
    try:
        from skopt import gp_minimize, space
        from skopt.utils import use_named_args
        
        search_space = [
            space.Integer(50, 95, name='min_confidence'),
            space.Integer(0, 50, name='max_confidence'),
            space.Real(1.0, 10.0, name='loss_threshold'),
            space.Real(0.0, 20.0, name='trail_vol_scale'),
            space.Integer(50, 2000, name='n_estimators'),
            space.Integer(50, 90, name='train_pct'),
            space.Integer(1, 49, name='val_pct'),
            space.Integer(0, 200, name='early_stopping'),
        ]
        try:
            X_all = pd.concat([X_train, X_test])
        except Exception:
            X_all = X_train.copy()
        
        @use_named_args(search_space)
        def objective(**params):
            try:
                min_conf_pct = float(params['min_confidence'])
                max_conf_pct = float(params['max_confidence'])
                loss_thresh_pct = float(params['loss_threshold'])
                trail_vol_pct = float(params['trail_vol_scale'])
                n_est = int(params['n_estimators'])
                train_pct_opt = int(params['train_pct'])
                val_pct_opt = int(params['val_pct'])
                
                if train_pct_opt + val_pct_opt >= 100:
                    return 1e6

                min_conf = min_conf_pct / 100.0
                max_conf = max_conf_pct / 100.0
                loss_thresh = loss_thresh_pct / 100.0
                trail_vol = trail_vol_pct / 100.0

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

                data_local = data.copy()
                data_local['Confidence'] = pd.Series(model.predict_proba(X_all)[:, 1], index=X_all.index)
                data_local['Signal'] = np.where(data_local['Confidence'] >= min_conf, 1, np.where(data_local['Confidence'] <= max_conf, -1, 0))
                data_local['Signal'] = data_local['Signal'].shift(1).fillna(0)
                
                bt_result = simulate_risk_aware_backtest(data_local, loss_thresh, trail_vol)
                data_local = bt_result['data']
                
                # Use only test period for evaluation
                strategy_returns = data_local.loc[X_test.index, 'PnL'].values if 'PnL' in data_local.columns else np.zeros(len(X_test))

                if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
                    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
                    return -sharpe
                else:
                    return 1e3
            except Exception as e:
                print(f"Error in objective function: {e}")
                return 1e3
        
        result = gp_minimize(objective, search_space, n_calls=max(3, int(n_calls)), random_state=42, n_initial_points=min(5, max(1, int(n_calls/3))))
        
        best_params = {
            'best_score': best_sharpe,
            'min_confidence': best_min_conf / 100.0, # Store as decimal for apply_bayesian_params
            'max_confidence': best_max_conf / 100.0, # Store as decimal for apply_bayesian_params
            'loss_threshold_pct': best_loss_threshold_pct / 100.0, # Store as decimal for apply_bayesian_params
            'trail_vol_scale': best_trail_vol_scale / 100.0, # Store as decimal for apply_bayesian_params
            'n_estimators': int(best_n_estimators),
            'train_pct': int(best_train_pct),
            'val_pct': int(best_val_pct),
            'early_stopping': int(best_early_stopping_rounds)
        }
        return best_params
    except Exception as e:
        print(f"Error in Bayesian optimization: {e}")
        return None

def _run_backtest_logic(symbol, start_date, end_date, interval, capital, min_confidence_long, max_confidence_short, loss_threshold_pct, trail_vol_scale, n_estimators, train_pct, val_pct, early_stopping_rounds, transaction_cost_bps, risk_free_rate_pct, progress_callback):
    try:
        print(f"DEBUG: _run_backtest_logic started for {symbol}")
        progress_callback(f"Fetching data for {symbol}...")
        loss_threshold = loss_threshold_pct / 100.0
        trail_vol_scale_val = trail_vol_scale / 100.0
        min_confidence = min_confidence_long / 100.0
        max_confidence = max_confidence_short / 100.0
        transaction_cost = (transaction_cost_bps if transaction_cost_bps else 10) / BPS_TO_DECIMAL
        risk_free_rate = (risk_free_rate_pct if risk_free_rate_pct else 5.0) / 100.0
        
        bars_per_year = get_bars_per_year(interval if interval else '1d')
        
        progress_callback("Calculating technical indicators...")
        strategy = Strategy()
        data = strategy.prepare_data(symbol if symbol else 'SPY', start_date, end_date, interval if interval else '1d')

        if data is None or data.empty:
            return html.Div([html.H4(f"Unable to fetch data for '{symbol}'" , style={'color': 'red'})], style={'textAlign': 'center', 'padding': '20px'})

        if len(data) < 100:
            return html.Div("Not enough data to run the backtest.", style={'color': 'red', 'textAlign': 'center'})

        progress_callback("Training XGBoost model...")
        X, y = strategy.get_feature_data(data)
        
        train_pct_val = int(train_pct) if train_pct else 65
        val_pct_val = int(val_pct) if val_pct else 15
        
        test_pct_val = 100 - train_pct_val - val_pct_val
        train_size = int(len(X) * (train_pct_val / 100.0))
        test_size = int(len(X) * (test_pct_val / 100.0))

        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:train_size+test_size], y[train_size:train_size+test_size]
        X_val, y_val = X[train_size+test_size:], y[train_size+test_size:]

        n_estimators_val = int(n_estimators) if n_estimators else 500
        esr = int(early_stopping_rounds) if early_stopping_rounds else 50

        model_xgb = strategy.train_model(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            n_estimators=n_estimators_val,
            early_stopping_rounds=esr
        )

        progress_callback("Running Cross-Validation...")
        cv_results = compute_cross_validation(model_xgb, X_train, y_train, n_splits=5)
            
        try:
            y_test_pred = model_xgb.predict(X_test)
            y_test_proba = model_xgb.predict_proba(X_test)[:, 1]
            cm = confusion_matrix(y_test, y_test_pred)
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_auc = auc(fpr, tpr)
        except Exception:
            cm, fpr, tpr, roc_auc = None, [], [], None

        progress_callback("Running Walk-Forward Validation...")
        y_proba_all = strategy.walk_forward_validation(X, y, train_size, n_estimators=n_estimators_val)
        
        data['Confidence'] = pd.Series(y_proba_all, index=data.index)
        data['Signal'] = np.where(data['Confidence'] > min_confidence, 1, 
                                  np.where(data['Confidence'] < max_confidence, -1, 0))
        data.loc[data['Confidence'].isna(), 'Signal'] = 0
        data['Signal'] = data['Signal'].shift(1)

        progress_callback("Simulating trades & risk...")
        bt_result = simulate_risk_aware_backtest(data, loss_threshold, trail_vol_scale_val)
        data = bt_result['data']
        trades_df = bt_result['trades']
        
        data['TradeEntry'] = (data['Signal'].diff().abs() > 0).astype(int)
        data['TransactionCosts'] = data['TradeEntry'] * transaction_cost * ENTRY_EXIT_COST_MULTIPLIER
        data['Returns'] = data['PnL'] * data['PositionSize'] - data['TransactionCosts']
        data['Cumulative Returns'] = (1 + data['Returns']).cumprod() * capital

        progress_callback("Generating charts & metrics...")
        final_value = data['Cumulative Returns'].iloc[-1] if not data['Cumulative Returns'].empty else capital
        total_return = (final_value / capital) - 1
        annualized_return = (1 + total_return) ** (bars_per_year / len(data)) - 1 if len(data) > 0 else 0
        daily_returns = data['Returns'].fillna(0)
        std_daily = daily_returns.std()
        
        volatility_annual = std_daily * np.sqrt(bars_per_year)
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility_annual if volatility_annual > 1e-6 else 0
        
        peak = data['Cumulative Returns'].cummax()
        drawdown = peak - data['Cumulative Returns']
        max_drawdown_val = drawdown.max()
        max_drawdown_pct = (max_drawdown_val / peak.max()) if peak.max() > 0 else 0
        
        # Precise Trade Metrics
        # trades_df already assigned from bt_result above, don't overwrite
        trade_metrics = calculate_trade_metrics(trades_df, capital)
        win_rate = trade_metrics['win_rate']
        profit_factor = trade_metrics['profit_factor']
        avg_win = trade_metrics['avg_win_pct']
        avg_loss = trade_metrics['avg_loss_pct']
        
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
Max Drawdown:              {max_drawdown_pct:.2%}
============================================================
Total Trades:              {trade_metrics['trades_count']}
Win Rate:                  {win_rate:.2%}
Profit Factor:             {profit_factor:.2f}
Avg Win:                   {avg_win:.2%}
Avg Loss:                  {avg_loss:.2%}
Exp. Value (per trade):    {trade_metrics['expectancy']:.2%}
============================================================
Transaction Costs Paid:    ${total_txn_costs:,.2f}
============================================================
⚠️  NOTE: Walk-forward validation used
    Continuously trained on expanding historical window
    Ensures all predictions are truly out-of-sample
============================================================
"""
        latest = data.iloc[-1]
        latest_signal = float(latest['Signal']) if hasattr(latest['Signal'], 'item') else 0
        latest_conf = float(latest['Confidence']) if hasattr(latest['Confidence'], 'item') else 0
        latest_signal_text = f"**Latest Signal**: {'Buy' if latest_signal == 1 else 'Sell' if latest_signal == -1 else 'Hold'} (Confidence: {latest_conf:.3f})"

        # --- Graph Styling ---
        graph_template = 'plotly_white'
        
        fig_price = px.line(data, x=data.index, y='Close', title=f'{symbol.upper()} Price & Trade Signals', template=graph_template)
        fig_price.update_traces(line_color='#42526e', line_width=1.5)
        fig_price.update_layout(height=800, hovermode='x unified', margin=dict(l=40, r=40, t=40, b=40))
        fig_price.add_scatter(x=data[data['Signal'] == 1].index, y=data[data['Signal'] == 1]['Close'], mode='markers', name='Buy Signal', marker=dict(color='#006644', size=10, symbol='triangle-up'))
        fig_price.add_scatter(x=data[data['Signal'] == -1].index, y=data[data['Signal'] == -1]['Close'], mode='markers', name='Sell Signal', marker=dict(color='#de350b', size=10, symbol='triangle-down'))
        fig_price.add_scatter(x=data[data['ExitReason'].str.contains('stop', na=False)].index, y=data[data['ExitReason'].str.contains('stop', na=False)]['Close'], mode='markers', name='Stop Trigger', marker=dict(color='#ffab00', size=8, symbol='x'))

        fig_portfolio = px.line(data, x=data.index, y='Cumulative Returns', title=f'Portfolio Value (Started with ${capital:,})', labels={'Cumulative Returns': 'Portfolio Value ($)'}, template=graph_template)
        fig_portfolio.update_traces(line_color='#0052cc', line_width=2)
        fig_portfolio.update_layout(height=600, hovermode='x unified', margin=dict(l=40, r=40, t=40, b=40))
        
        fig_returns = px.histogram(data, x='Returns', nbins=100, title='Strategy Daily Returns Distribution', labels={'Returns': 'Daily Return'}, template=graph_template)
        fig_returns.update_traces(marker_color='#0052cc', opacity=0.7)
        fig_returns.update_layout(height=500, hovermode='x unified', margin=dict(l=40, r=40, t=40, b=40))

        # Persist run
        try:
            runs_dir = os.path.join(os.getcwd(), 'runs')
            os.makedirs(runs_dir, exist_ok=True)
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(runs_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            joblib.dump(model_xgb, os.path.join(run_dir, 'model.joblib'))
            data.to_csv(os.path.join(run_dir, 'backtest_results.csv'))
            saved_run_path = run_dir
        except Exception:
            saved_run_path = None

        extra_graphs = []
        if cm is not None:
            fig_cm = go.Figure(data=go.Heatmap(z=cm.tolist(), x=['Pred 0','Pred 1'], y=['True 0','True 1'], colorscale='Blues'))
            fig_cm.update_layout(title='Confusion Matrix', template=graph_template)
            extra_graphs.append(dcc.Graph(figure=fig_cm))
        if roc_auc is not None:
            fig_roc = go.Figure()
            fig_roc.add_scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})', line_color='#0052cc')
            fig_roc.add_scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='#97a0af'), name='Chance')
            fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', template=graph_template)
            extra_graphs.append(dcc.Graph(figure=fig_roc))

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
            html.Div(className='metric-card', children=[
                html.Div(className='metric-label', children='CV Score (Train)'),
                html.Div(className='metric-value neutral', children=f'{cv_results["mean_score"]:.3f}' if cv_results else 'N/A'),
                html.Div(className='metric-change', children=f'±{cv_results["std_score"]:.3f} (5-fold)' if cv_results else 'on train set')
            ]),
        ])

        progress_callback("Analyzing Factors & Risk Parity...")
        rp_analysis = analyze_risk_parity(data)
        factor_attr = compute_factor_attribution(data)
        
        rp_viz_div = html.Div()
        if rp_analysis:
            rp_weights = rp_analysis['weights'].sort_values(ascending=False)
            fig_rp = go.Figure(data=[go.Bar(x=rp_weights.index, y=rp_weights.values, marker_color='#42526e', text=[f'{v:.4f}' for v in rp_weights.values], textposition='auto')])
            fig_rp.update_layout(title='Risk Parity Factor Weights', height=400, template=graph_template)
            
            factor_vols = rp_analysis.get('factor_vols', {})
            sig_vols = {k: v for k, v in factor_vols.items() if v > 1e-8}
            if sig_vols:
                fig_contrib = go.Figure(data=[go.Pie(labels=list(sig_vols.keys()), values=list(sig_vols.values()), marker=dict(colors=px.colors.qualitative.Prism), textinfo='label+percent')])
                fig_contrib.update_layout(title='Factor Volatility Contribution', height=400, template=graph_template)
            else:
                fig_contrib = go.Figure()

            rp_viz_div = html.Div([dcc.Graph(figure=fig_rp), dcc.Graph(figure=fig_contrib)], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'})
        
        factor_attr_div = html.Div()
        if factor_attr:
            sorted_factors = factor_attr['sorted_factors']
            fig_corr = go.Figure(data=[go.Bar(x=[f[0] for f in sorted_factors], y=[f[1] for f in sorted_factors], marker_color=['#006644' if f[1] > 0 else '#de350b' for f in sorted_factors])])
            fig_corr.update_layout(title='Factor Correlation with Strategy Returns', xaxis_title='Factor', yaxis_title='Correlation', template=graph_template)
            
            top_3_text = "Top Factors Contributing to Returns:\n"
            for i, (factor, corr) in enumerate(factor_attr['top_3_factors'][:3], 1):
                top_3_text += f"\n{i}. {factor}: {corr:.3f} correlation"
            
            factor_attr_div = html.Div([dcc.Graph(figure=fig_corr), html.Pre(top_3_text, style={'backgroundColor': '#f4f5f7', 'padding': '12px', 'borderRadius': '3px', 'fontSize': '12px'})])

        # --- Feature Importance ---
        importance_div = html.Div()
        try:
            if hasattr(model_xgb, 'feature_importances_'):
                fi = model_xgb.feature_importances_
                fi_df = pd.DataFrame({'Feature': features, 'Importance': fi}).sort_values('Importance', ascending=True)
                
                fig_feat = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Feature Importance (XGBoost)', template=graph_template)
                fig_feat.update_traces(marker_color='#36b37e')
                fig_feat.update_layout(height=400)
                importance_div = dcc.Graph(figure=fig_feat)
        except Exception:
            importance_div = html.Div("Feature importance not available.")

        # --- Trade Log Table ---
        trade_log_div = html.Div()
        if trades_df is not None and not trades_df.empty:
            # Format for display
            disp_trades = trades_df.copy()
            disp_trades['PnL'] = disp_trades['PnL'].apply(lambda x: f"{x:.2%}")
            disp_trades['EntryPrice'] = disp_trades['EntryPrice'].apply(lambda x: f"{x:.2f}")
            disp_trades['ExitPrice'] = disp_trades['ExitPrice'].apply(lambda x: f"{x:.2f}")
            disp_trades['EntryDate'] = disp_trades['EntryDate'].dt.strftime('%Y-%m-%d')
            disp_trades['ExitDate'] = disp_trades['ExitDate'].dt.strftime('%Y-%m-%d')
            
            # Select cols
            disp_cols = ['EntryDate', 'ExitDate', 'EntryPrice', 'ExitPrice', 'PnL', 'ExitReason']
            disp_trades = disp_trades[disp_cols]
            
            trade_log_div = html.Div([
                html.Div(className='section-header', children=[html.H3('Detailed Trade Log')]),
                dash_table.DataTable(
                    data=disp_trades.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in disp_trades.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#f4f6f8', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{PnL} contains "-"'},
                            'color': '#de350b', 'fontWeight': 'bold'
                        },
                        {
                            'if': {'filter_query': '{PnL} contains "+"'},
                            'color': '#006644', 'fontWeight': 'bold'
                        }
                    ]
                )
            ])

        results_sections = [
            html.Div(className='results-section', children=[html.Div(className='section-header', children=[html.H3('Performance Metrics')]), metric_cards]),
            html.Div(className='results-section', children=[html.Div(className='section-header', children=[html.H3('Price & Signals')]), dcc.Graph(figure=fig_price)]),
            html.Div(className='results-section', children=[html.Div(className='section-header', children=[html.H3('Portfolio Value')]), dcc.Graph(figure=fig_portfolio)]),
            html.Div(className='results-section', children=[html.Div(className='section-header', children=[html.H3('Daily Returns Distribution')]), dcc.Graph(figure=fig_returns)]),
            html.Div(className='results-section', children=[html.Div(className='section-header', children=[html.H3('Model Insights')]), importance_div]),
            html.Div(className='results-section', children=[trade_log_div]),
        ]
        
        if rp_analysis:
            results_sections.append(html.Div(className='results-section', children=[html.Div(className='section-header', children=[html.H3('Risk Parity Analysis')]), rp_viz_div]))
        
        if factor_attr:
            results_sections.append(html.Div(className='results-section', children=[html.Div(className='section-header', children=[html.H3('Factor Attribution Analysis')]), factor_attr_div]))
        
        if cm is not None or roc_auc is not None:
            results_sections.append(html.Div(className='results-section', children=[html.Div(className='section-header', children=[html.H3('Model Diagnostics')]), html.Div(extra_graphs, style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'})]))
        
        results_sections.append(html.Div(className='results-section', children=[
            html.Div(className='section-header', children=[html.H3('Execution Summary')]),
            html.Pre(summary_text, className='summary-container'),
            html.H5(latest_signal_text, style={'textAlign': 'center', 'fontWeight': 'bold', 'marginTop': '16px'})
        ]))

        # Prepare Data for Store
        metrics_data = {
            'total_return': total_return,
            'sharpe': sharpe_ratio,
            'max_drawdown': max_drawdown_pct,
            'win_rate': win_rate,
            'trades': int((data['PnL'] != 0).sum()),
            'annual_vol': volatility_annual,
            'final_value': final_value
        }
        
        # Equity Curve (Resampled to limit size if needed, but daily is fine)
        # Normalize Benchmark (Close) to start at same capital or 1.0
        initial_close = data['Close'].iloc[0]
        benchmark_curve = (data['Close'] / initial_close) * capital
        
        equity_curve_data = []
        for idx, row in data.iterrows():
            equity_curve_data.append({
                'Date': idx.strftime('%Y-%m-%d'),
                'Strategy': row['Cumulative Returns'],
                'Benchmark': benchmark_curve.loc[idx]
            })

        return {
            'html': html.Div(results_sections, style={'width': '100%'}),
            'metrics': metrics_data,
            'equity_curve': equity_curve_data
        }

    except Exception as e:
        error_html = html.Div([
            html.H4("An error occurred:", style={'color': 'red'}),
            html.Pre(f"{traceback.format_exc()}", style={'border': '1px solid #ddd', 'padding': '10px', 'overflowX': 'auto'})
        ])
        return {'html': error_html, 'metrics': None, 'equity_curve': None}

def _run_backtest_bg(args):
    global backtest_status
    print("DEBUG: Background thread started")
    
    def update_progress(msg):
        with backtest_lock:
            backtest_status['progress'] = msg
            
    try:
        print("DEBUG: Calling _run_backtest_logic")
        result_data = _run_backtest_logic(*args, update_progress)
        print("DEBUG: Logic finished, result obtained")
        with backtest_lock:
            backtest_status['result'] = result_data
    except Exception as e:
        print(f"DEBUG: Exception in background thread: {e}")
        print(traceback.format_exc())
        with backtest_lock:
            backtest_status['result'] = {'html': html.Div(f"Critical Error: {e}", style={'color': 'red'}), 'metrics': None, 'equity_curve': None}
    finally:
        with backtest_lock:
            backtest_status['running'] = False
        print("DEBUG: Background thread finished")

@app.callback(
    [Output('backtest-interval', 'disabled'),
     Output('results-output', 'children'),
     Output('run-button', 'disabled'),
     Output('current-run-store', 'data'),
     Output('btn-lock-baseline', 'disabled')],
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
     State('input-risk-free-rate', 'value')],
    prevent_initial_call=True
)
def start_backtest(n_clicks, *args):
    print(f"DEBUG: Run Backtest clicked. n_clicks={n_clicks}", flush=True)
    if n_clicks == 0:
        return True, html.Div("Set parameters and click 'Run Backtest' to start.", style={'textAlign': 'center', 'marginTop': '50px'}), False, no_update, True
    
    global backtest_status
    with backtest_lock:
        if backtest_status['running']:
            print("DEBUG: Backtest already running.")
            return no_update, no_update, True, no_update, True
        backtest_status['running'] = True
        backtest_status['progress'] = 'Initializing...'
        backtest_status['result'] = None
    
    print("DEBUG: Starting background thread")
    thread = threading.Thread(target=_run_backtest_bg, args=(args,))
    thread.daemon = True
    thread.start()
    
    # Return disabled=False for interval, show loading UI immediately
    loading_ui = html.Div(className='loading-container', children=[
        html.Div(className='loader'),
        html.Div("Initializing Strategy Engine...", className='loading-text'),
        html.Div("Preparing data environment...", className='loading-subtext')
    ])
    
    return False, loading_ui, True, no_update, True

@app.callback(
    [Output('backtest-interval', 'disabled', allow_duplicate=True),
     Output('results-output', 'children', allow_duplicate=True),
     Output('run-button', 'disabled', allow_duplicate=True),
     Output('current-run-store', 'data', allow_duplicate=True),
     Output('btn-lock-baseline', 'disabled', allow_duplicate=True)],
    Input('backtest-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_backtest_progress(n):
    global backtest_status
    
    with backtest_lock:
        is_running = backtest_status['running']
        progress_msg = backtest_status['progress']
        result = backtest_status['result']
    
    if n % 5 == 0: # Reduce log spam
        print(f"DEBUG: Interval fired. Running={is_running}, Msg={progress_msg}, Result={'Found' if result else 'None'}")

    if result is not None:
        print("DEBUG: Backtest complete. Returning results.")
        # Done
        html_content = result.get('html')
        metrics = result.get('metrics')
        curve = result.get('equity_curve')
        
        store_data = {'metrics': metrics, 'equity_curve': curve} if metrics else None
        btn_disabled = False if metrics else True
        
        return True, html_content, False, store_data, btn_disabled
    
    if is_running:
        # Update progress message
        loading_ui = html.Div(className='loading-container', children=[
            html.Div(className='loader'),
            html.Div(progress_msg, className='loading-text'),
            html.Div("System processing...", className='loading-subtext')
        ])
        return no_update, loading_ui, no_update, no_update, True
    
    return True, no_update, False, no_update, True

@app.callback(
    [Output('baseline-run-store', 'data'),
     Output('btn-lock-baseline', 'children'),
     Output('baseline-status', 'children')],
    Input('btn-lock-baseline', 'n_clicks'),
    State('current-run-store', 'data'),
    prevent_initial_call=True
)
def lock_baseline(n_clicks, current_data):
    if not current_data or not n_clicks:
        return no_update, "Set Current as Baseline", ""
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    return current_data, "Update Baseline", f"✓ Baseline set at {timestamp}"

@app.callback(
    Output('comparison-output', 'children'),
    [Input('current-run-store', 'data'),
     Input('baseline-run-store', 'data')]
)
def render_comparison(current, baseline):
    if not current or not baseline:
        return None
        
    # Extract metrics
    m_curr = current['metrics']
    m_base = baseline['metrics']
    
    # Helpers for formatting
    def fmt_pct(val): return f"{val*100:+.2f}%"
    def fmt_num(val): return f"{val:.3f}"
    
    # Deltas
    d_ret = m_curr['total_return'] - m_base['total_return']
    d_sharpe = m_curr['sharpe'] - m_base['sharpe']
    d_dd = m_curr['max_drawdown'] - m_base['max_drawdown'] # Lower is better usually, but here we just show diff
    d_vol = m_curr['annual_vol'] - m_base['annual_vol']
    
    def get_arrow(val, inverse=False):
        if abs(val) < 0.0001: return "—", "delta-neutral"
        is_good = val > 0 if not inverse else val < 0
        symbol = "▲" if val > 0 else "▼"
        cls = "delta-positive" if is_good else "delta-negative"
        return symbol, cls

    # Scorecard HTML construction
    rows = [
        ("Total Return", fmt_pct(m_base['total_return']), fmt_pct(m_curr['total_return']), d_ret, False, fmt_pct),
        ("Sharpe Ratio", fmt_num(m_base['sharpe']), fmt_num(m_curr['sharpe']), d_sharpe, False, fmt_num),
        ("Max Drawdown", fmt_pct(m_base['max_drawdown']), fmt_pct(m_curr['max_drawdown']), d_dd, True, fmt_pct), # Inverse: lower DD is better
        ("Annual Volatility", fmt_pct(m_base['annual_vol']), fmt_pct(m_curr['annual_vol']), d_vol, True, fmt_pct), # Inverse: lower Vol is better
    ]
    
    scorecard_divs = []
    for label, b_val, c_val, delta, inv, fmt_func in rows:
        arrow, cls = get_arrow(delta, inv)
        delta_str = f"{arrow} {fmt_func(delta)}"
        scorecard_divs.append(html.Div(className='scorecard-row', children=[
            html.Div(label, className='scorecard-cell scorecard-label'),
            html.Div(b_val, className='scorecard-cell'),
            html.Div(c_val, className='scorecard-cell'),
            html.Div(delta_str, className=f'scorecard-cell {cls}')
        ]))
        
    scorecard_header = html.Div(className='scorecard-row', children=[
        html.Div("METRIC", className='scorecard-header'),
        html.Div("BASELINE (A)", className='scorecard-header'),
        html.Div("CURRENT (B)", className='scorecard-header'),
        html.Div("DELTA (B-A)", className='scorecard-header'),
    ])
    
    # Comparison Chart
    df_curr = pd.DataFrame(current['equity_curve'])
    df_base = pd.DataFrame(baseline['equity_curve'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_base['Date'], y=df_base['Strategy'], name='Baseline (A)', line=dict(color='#97a0af', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=df_curr['Date'], y=df_curr['Strategy'], name='Current (B)', line=dict(color='#0052cc', width=2)))
    
    # Add benchmark from current run for context
    fig.add_trace(go.Scatter(x=df_curr['Date'], y=df_curr['Benchmark'], name='Buy & Hold', line=dict(color='#42526e', width=1), opacity=0.5))
    
    fig.update_layout(
        title='Equity Curve Comparison: Baseline vs. Current',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return html.Div(className='results-section', children=[
        html.Div(className='section-header', children=[html.H3('Comparative Analysis (A/B Testing)')]),
        html.Div(className='comparison-scorecard', children=[scorecard_header] + scorecard_divs),
        dcc.Graph(figure=fig)
    ])

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
            # Calculate ETA
            completed = exp_status['completed_trials']
            total = exp_status['total_trials']
            start = exp_status['start_time']
            eta_str = "Estimating..."
            
            if completed > 0 and start:
                elapsed = time.time() - start
                avg_time = elapsed / completed
                remaining = total - completed
                eta_seconds = remaining * avg_time
                
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                else:
                    eta_str = f"{eta_seconds/60:.1f}m"
            
            msg = exp_status['progress']
            
            return html.Div(className='loading-container', children=[
                html.Div(className='loader'),
                html.Div(msg, className='loading-text'),
                html.Div(f"Completed: {completed}/{total} | ETA: {eta_str}", className='loading-subtext')
            ])
        else:
            return exp_status['progress']

@app.callback(
    [Output('bayes-progress', 'children'),
     Output('bayes-best-params', 'data'),
     Output('bayes-results', 'children'),
     Output('bayes-apply-button', 'disabled')],
    Input('bayes-opt-button', 'n_clicks'),
    [State('input-symbol', 'value'),
     State('input-start-date', 'value'),
     State('input-end-date', 'value'),
     State('input-interval', 'value')],
    prevent_initial_call=True
)
def run_bayesian_optimization(n_clicks, symbol, start_date, end_date, interval):
    if not n_clicks or not BAYESIAN_OPT_AVAILABLE:
        return 'Bayesian optimization requires scikit-optimize. Install: pip install scikit-optimize', None, '', True
    
    try:
        # Fetch and prepare data
        bars_per_year = get_bars_per_year(interval if interval else '1d')
        data = fetch_stock_data(symbol if symbol else 'SPY', start_date, end_date, interval if interval else '1d')
        
        if data is None or data.empty:
            return f'ERROR: Unable to fetch data for {symbol}', None, '', True
        
        # Prepare features
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
        
        # Split data
        train_size = int(len(X_all) * 0.7)
        X_train = X_all[:train_size]
        y_train = y_all[:train_size]
        X_test = X_all[train_size:]
        y_test = y_all[train_size:]
        
        # Run optimization
        best_params = bayesian_optimize_strategy(data, X_train, y_train, X_test, y_test, n_calls=20)
        
        if best_params is None:
            return 'Optimization failed or scikit-optimize not available.', None, '', True
        
                # Format results
        
                results_html = html.Div([
        
                    html.P(f"✓ Optimization complete! Best Sharpe: {best_params['best_score']:.3f}"),
        
                    html.Ul([
        
                        html.Li(f"Min Confidence: {best_params['min_confidence']*100:.0f}%"),
        
                        html.Li(f"Max Confidence: {best_params['max_confidence']*100:.0f}%"),
        
                        html.Li(f"Loss Threshold: {best_params['loss_threshold_pct']*100:.2f}%"),
        
                        html.Li(f"Trail Vol Scale: {best_params['trail_vol_scale']*100:.2f}%"),
        
                        html.Li(f"N Estimators: {best_params['n_estimators']}"),
        
                        html.Li(f"Train %: {best_params['train_pct']}% "),
        
                        html.Li(f"Val %: {best_params['val_pct']}% "),
        
                        html.Li(f"Early Stopping: {best_params['early_stopping']}"),
        
                    ])
        
                ])
        
        return 'Optimization complete! Click "Apply Optimized Params" to use these settings.', best_params, results_html, False
        
    except Exception as e:
        return f'ERROR: {str(e)}', None, '', True

@app.callback(
    [Output('slider-min-confidence', 'value'),
     Output('slider-max-confidence', 'value'),
     Output('slider-loss-threshold', 'value'),
     Output('slider-trail-vol-scale', 'value'),
     Output('input-n-estimators', 'value'),
     Output('slider-train-pct', 'value'),
     Output('slider-val-pct', 'value'),
     Output('input-early-stopping', 'value')],
    Input('bayes-apply-button', 'n_clicks'),
    State('bayes-best-params', 'data'),
    prevent_initial_call=True
)
def apply_bayesian_params(n_clicks, best_params):
    if not n_clicks or not best_params:
        raise dash.exceptions.PreventUpdate
    
    # Rounding and converting to match UI component expectations
    min_conf_out = round(best_params['min_confidence'] * 100) if best_params['min_confidence'] is not None else no_update
    max_conf_out = round(best_params['max_confidence'] * 100) if best_params['max_confidence'] is not None else no_update
    loss_thresh_out = round(best_params['loss_threshold_pct'] * 100, 2) if best_params['loss_threshold_pct'] is not None else no_update
    trail_vol_scale_out = round(best_params['trail_vol_scale'] * 100, 2) if best_params['trail_vol_scale'] is not None else no_update
    n_estimators_out = round(best_params['n_estimators']) if best_params['n_estimators'] is not None else no_update
    train_pct_out = round(best_params['train_pct']) if best_params['train_pct'] is not None else no_update
    val_pct_out = round(best_params['val_pct']) if best_params['val_pct'] is not None else no_update
    early_stopping_out = round(best_params['early_stopping']) if best_params['early_stopping'] is not None else no_update

    return (
        min_conf_out,
        max_conf_out,
        loss_thresh_out,
        trail_vol_scale_out,
        n_estimators_out,
        train_pct_out,
        val_pct_out,
        early_stopping_out
    )

@app.callback(
    Output('exp-progress', 'children'),
    Input('run-experiments-button', 'n_clicks'),
    [State('exp-n-estimators', 'value'), State('exp-train-pct', 'value'), State('exp-val-pct', 'value'), State('exp-early-stopping', 'value'), State('exp-mode', 'value'), State('exp-n-trials', 'value'), State('input-symbol','value'), State('input-start-date','value'), State('input-end-date','value'), State('input-interval','value'), State('slider-min-confidence','value'), State('slider-max-confidence','value'), State('slider-loss-threshold','value'), State('slider-trail-vol-scale','value')],
    prevent_initial_call=True
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
        strategy = Strategy()
        data = strategy.prepare_data(symbol if symbol else 'SPY', start_date, end_date, interval if interval else '1d')

        if data is None or data.empty:
            update_progress(f'ERROR: Unable to fetch data for {symbol}. Yahoo Finance API may be temporarily unavailable. Please try again.')
            with exp_lock:
                exp_status['running'] = False
            return
        
        X_all, y_all = strategy.get_feature_data(data)

        update_progress('Building parameter combinations...')
        combos = []
        if mode == 'grid':
            for ne, tr, va, es in itertools.product(n_est_list, train_list, val_list, es_list if es_list else [0]):
                combos.append({'n_estimators':ne, 'train_pct':tr, 'val_pct':va, 'early_stopping':es})
        else:
            import random
            for _ in range(int(n_trials) if n_trials else 10):
                combos.append({'n_estimators': random.choice(n_est_list), 'train_pct': random.choice(train_list), 'val_pct': random.choice(val_list), 'early_stopping': random.choice(es_list) if es_list else 0})

        with exp_lock:
            exp_status['total_trials'] = len(combos)
            exp_status['completed_trials'] = 0
            exp_status['start_time'] = time.time()

        results = []
        runs_dir = os.path.join(os.getcwd(), 'experiments')
        os.makedirs(runs_dir, exist_ok=True)
        exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_file = os.path.join(runs_dir, f'exp_{exp_id}.csv')

        for idx, combo in enumerate(combos):
            # ETA Calc happens in poll_exp_progress based on completed count
            update_progress(f'Running trial {idx+1}/{len(combos)}')
            
            ne = combo['n_estimators']
            tr = combo['train_pct']
            va = combo['val_pct'] # ignored in walk-forward, but kept for structure
            es = combo['early_stopping'] # ignored in this simple walk-forward implementation unless extended
            
            # In Walk-Forward, "train_pct" defines the initial window size
            initial_train_size = int(len(X_all) * (tr/100.0))
            if initial_train_size < 50: # Min window
                 initial_train_size = 50
            
            try:
                # Use Walk-Forward Validation for robustness
                # Note: This is slower than simple split but much more accurate
                y_proba_all = strategy.walk_forward_validation(X_all, y_all, train_size=initial_train_size, n_estimators=int(ne))
            except Exception:
                with exp_lock:
                    exp_status['completed_trials'] += 1
                continue

            # Create test data subset (the part that has predictions)
            # walk_forward_validation returns NaN for the initial window
            valid_indices = ~np.isnan(y_proba_all)
            
            if not np.any(valid_indices):
                 with exp_lock:
                    exp_status['completed_trials'] += 1
                 continue
                 
            y_test = y_all[valid_indices]
            y_proba_test = y_proba_all[valid_indices]
            y_test_pred = (y_proba_test > 0.5).astype(int) # Simple threshold for accuracy metric

            # Evaluate Classification Metrics
            try:
                cm = confusion_matrix(y_test, y_test_pred)
                fpr, tpr, _ = roc_curve(y_test, y_proba_test)
                roc_auc = auc(fpr,tpr)
                accuracy = (y_test_pred==y_test).mean()
            except Exception:
                cm = None; roc_auc=0; accuracy=0

            # Run Backtest on these Out-of-Sample predictions
            data_test = data.loc[valid_indices].copy()
            data_test['Confidence'] = y_proba_test
            data_test['Signal'] = np.where(data_test['Confidence'] >= min_conf/100.0, 1, 
                                           np.where(data_test['Confidence'] <= max_conf/100.0, -1, 0))
            data_test['Signal'] = data_test['Signal'].shift(1).fillna(0)
            
            # Run backtest
            res_local = simulate_risk_aware_backtest(data_test, loss_threshold_pct/100.0, trail_vol_scale)
            
            # Calculate metrics
            returns = res_local['PnL'].copy()
            final_value = (1 + returns).cumprod().iloc[-1] if not returns.empty and len(returns) > 0 else 1.0
            
            adv_metrics = calculate_advanced_metrics(returns)
            trades_df = getattr(res_local, 'trades', None)
            trade_metrics = calculate_trade_metrics(trades_df)
            
            results.append({
                'n_estimators': ne, 
                'train_pct': tr, 
                'val_pct': va, 
                'early_stopping': es, 
                'roc_auc': roc_auc if roc_auc else 0, 
                'accuracy': accuracy if accuracy else 0, 
                'sharpe_ratio': adv_metrics.get('sharpe', 0),
                'sortino_ratio': adv_metrics.get('sortino', 0),
                'calmar_ratio': adv_metrics.get('calmar', 0),
                'max_drawdown': adv_metrics.get('max_dd', 0),
                'profit_factor': trade_metrics.get('profit_factor', 0),
                'win_rate': trade_metrics.get('win_rate', 0),
                'final_value': float(final_value)
            })
            
            with exp_lock:
                exp_status['completed_trials'] += 1

        # save results
        try:
            df_res = pd.DataFrame(results)
            
            # CSV Save
            df_res.to_csv(exp_file, index=False)
            
            # Generate Professional HTML Report
            report_file = os.path.join(runs_dir, f'report_{exp_id}.html')
            
            # 1. Parallel Coords
            fig_par = px.parallel_coordinates(
                df_res, 
                color="sharpe_ratio", 
                dimensions=['n_estimators', 'train_pct', 'sharpe_ratio', 'profit_factor', 'win_rate', 'max_drawdown'],
                color_continuous_scale=px.colors.diverging.Tealrose,
                title="Hyperparameter Efficiency Frontier (Walk-Forward Validation)"
            )
            fig_par.update_layout(template='plotly_white')
            
            # 2. Risk-Reward Scatter
            fig_scatter = px.scatter(
                df_res, x='volatility', y='sharpe_ratio', 
                size='final_value', color='max_drawdown',
                hover_data=['n_estimators', 'train_pct'],
                title="Risk vs. Reward Landscape",
                labels={'volatility': 'Annualized Volatility', 'sharpe_ratio': 'Sharpe Ratio'}
            )
            fig_scatter.update_layout(template='plotly_white')

            # 3. Generate HTML
            with open(report_file, 'w') as f:
                f.write(f"""
                <html>
                <head>
                    <title>Strategy Experiment Report - {exp_id}</title>
                    <style>
                        body {{ font-family: -apple-system, sans-serif; padding: 40px; background: #f4f6f8; color: #172b4d; }}
                        .container {{ max-width: 100%; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                        h1 {{ border-bottom: 2px solid #0052cc; padding-bottom: 10px; }}
                        .table-wrapper {{ overflow-x: auto; margin-top: 20px; }}
                        table {{ width: 100%; border-collapse: collapse; min-width: 1400px; }}
                        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ebecf0; white-space: nowrap; font-size: 13px; }}
                        th {{ background-color: #f4f5f7; font-weight: 600; color: #42526e; position: sticky; top: 0; }}
                        td {{ font-family: 'Consolas', monospace; }}
                        .highlight {{ color: #006644; font-weight: bold; }}
                        tbody tr:hover {{ background-color: #f8f9fa; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Quantitative Experiment Report</h1>
                        <p><strong>Run ID:</strong> {exp_id}<br><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                        
                        <h2>1. Strategy Efficiency</h2>
                        {fig_par.to_html(full_html=False, include_plotlyjs='cdn')}
                        
                        <h2>2. Risk/Reward Analysis</h2>
                        {fig_scatter.to_html(full_html=False, include_plotlyjs=False)}
                        
                        <h2>3. Top Performing Configurations</h2>
                        <div class="table-wrapper">
                            {df_res.sort_values('sharpe_ratio', ascending=False).head(10).to_html(classes='table', border=0, float_format='%.6f')}
                        </div>
                    </div>
                </body>
                </html>
                """)
                
        except Exception as e:
            print(f"Report generation failed: {e}")

        # build results for dashboard
        if not results:
            update_progress('No valid experiment runs completed.')
        else:
            # Interactive Table
            table = dash_table.DataTable(
                data=df_res.to_dict('records'),
                columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ".3f"}} for i in df_res.columns],
                sort_action="native",
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#f4f6f8', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'left', 'padding': '10px'}
            )
            
            result_div = html.Div([
                html.H5('Experiment Results'), 
                dcc.Graph(figure=fig_par),
                html.Br(),
                html.H6('Detailed Metrics Table'),
                table, 
                html.Div([
                    html.Span(f'Data saved to: {exp_file}'),
                    html.Br(),
                    html.B(f'Professional Report Generated: report_{exp_id}.html', style={'color': '#0052cc'})
                ], style={'fontStyle':'italic', 'marginTop': '10px', 'backgroundColor': '#e3f2fd', 'padding': '10px', 'borderRadius': '4px'})
            ])
            
            with exp_lock:
                exp_status['results'] = result_div
                exp_status['progress'] = f'Completed {len(results)} runs.'

    except Exception as e:
        update_progress(f'ERROR: {str(e)}')
        print(traceback.format_exc())
    finally:
        with exp_lock:
            exp_status['running'] = False

if __name__ == '__main__':
    # Enable debug mode to show errors in the browser and console
    app.run(debug=True, host='127.0.0.1', port=8050, use_reloader=False)
