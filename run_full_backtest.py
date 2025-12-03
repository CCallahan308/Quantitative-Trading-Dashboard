#!/usr/bin/env python3
"""Headless runner to execute the full backtest and print/save results."""
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px
import numpy as np
import os

from core.strategy import Strategy
from core.backtest import simulate_risk_aware_backtest
from core.data import get_bars_per_year
from core.utils import compute_cross_validation

# Constants
BPS_TO_DECIMAL = 10000.0  # Convert basis points to decimal
ENTRY_EXIT_COST_MULTIPLIER = 2  # Transaction cost applied on both entry and exit


def run(symbol='SPY', start_date='2024-01-01', end_date=None, interval='1d', capital=1000, min_conf=0.55, max_conf=0.45, loss_threshold_pct=5, trail_vol_scale=2.0, transaction_cost_bps=10, risk_free_rate_pct=5.0):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    loss_threshold = loss_threshold_pct / 100.0
    transaction_cost = transaction_cost_bps / BPS_TO_DECIMAL
    risk_free_rate = risk_free_rate_pct / 100.0  # Convert percentage to decimal
    
    # Calculate bars per year based on interval
    bars_per_year = get_bars_per_year(interval)

    print(f"Fetching data and generating features for {symbol}...")
    strategy = Strategy()
    data = strategy.prepare_data(symbol, start_date, end_date, interval)
    
    if data is None or data.empty:
        print('No data returned')
        return

    X, y = strategy.get_feature_data(data)

    # Load model training splits
    train_size = int(len(X) * 0.65)
    test_size = int(len(X) * 0.20)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:train_size+test_size]
    y_test = y[train_size:train_size+test_size]

    # Train model
    print("Training model...")
    
    # Check CV first
    # Create a temporary model for CV (since train_model fits it)
    # We use the same params as default in Strategy.train_model
    tmp_model = strategy.train_model(X_train, y_train, n_estimators=10) # Just to get a model instance? No, train_model returns a fitted model.
    # We need a fresh unfitted model for CV or just pass the fitted one if compute_cross_validation handles cloning.
    # Looking at core/utils.py (implied), usually sklearn cloning is safe.
    # But let's just train the main model and skip CV print for now to keep it simple, 
    # or use the method from strategy if I added one (I didn't).
    # Let's stick to the original logic but using strategy.train_model
    
    model = strategy.train_model(X_train, y_train, X_val=X_test, y_val=y_test, early_stopping_rounds=50)

    # Perform Walk-Forward Validation
    print("Running Walk-Forward Validation...")
    y_proba_all = strategy.walk_forward_validation(X, y, train_size)
    
    data['Confidence'] = pd.Series(y_proba_all, index=X.index)
    data['Signal'] = np.where(data['Confidence'] > min_conf, 1, 
                              np.where(data['Confidence'] < max_conf, -1, 0))
    # Don't generate signals for training period (where confidence is NaN)
    data.loc[data['Confidence'].isna(), 'Signal'] = 0
    data['Signal'] = data['Signal'].shift(1)

    # Backtest with transaction costs
    print("Running backtest...")
    bt_result = simulate_risk_aware_backtest(data, loss_threshold, trail_vol_scale)
    res = bt_result['data']
    trades_df = bt_result['trades']
    
    # Calculate returns with transaction costs
    res['TradeEntry'] = (res['Signal'].diff().abs() > 0).astype(int)
    res['TransactionCosts'] = res['TradeEntry'] * transaction_cost * ENTRY_EXIT_COST_MULTIPLIER
    res['Returns'] = res['PnL'] * res['PositionSize'] - res['TransactionCosts']
    res['Cumulative Returns'] = (1 + res['Returns']).cumprod() * capital

    final_value = res['Cumulative Returns'].iloc[-1] if not res['Cumulative Returns'].empty else capital
    total_return = (final_value / capital) - 1
    annualized_return = (1 + total_return) ** (bars_per_year / len(res)) - 1 if len(res)>0 else 0
    daily_returns = res['Returns'].fillna(0)
    std_daily = daily_returns.std()
    
    # Calculate proper Sharpe ratio with risk-free rate
    volatility_annual = std_daily * (bars_per_year ** 0.5)
    excess_return = annualized_return - risk_free_rate
    sharpe = excess_return / volatility_annual if volatility_annual > 1e-6 else 0
    
    win_rate = (res['Returns'] > 0).sum() / (res['PnL'] != 0).sum() if (res['PnL'] != 0).sum() > 0 else 0
    trades = (res['PnL'] != 0).sum()
    total_txn_costs = res['TransactionCosts'].sum() * capital

    print('--- Backtest Summary ---')
    print(f'Symbol: {symbol}  Interval: {interval}  Period: {start_date} to {end_date}')
    print(f'Final Value: ${final_value:,.2f}')
    print(f'Total Return: {total_return:.2%}  Annualized: {annualized_return:.2%}')
    print(f'Annual Volatility: {volatility_annual:.2%}')
    print(f'Sharpe Ratio: {sharpe:.2f} (Risk-Free Rate: {risk_free_rate:.2%})')
    print(f'Win rate: {win_rate:.2%}  Trades: {trades}')
    print(f'Transaction Costs Paid: ${total_txn_costs:,.2f}')
    print('')
    print('NOTE: Sharpe calculated on OUT-OF-SAMPLE data (Walk-Forward Validation)')

    # Save charts
    fig_price = px.line(res, x=res.index, y='Close', title=f'{symbol} Price & Signals')
    fig_price.add_scatter(x=res[res['Signal']==1].index, y=res[res['Signal']==1]['Close'], mode='markers', name='Buy', marker=dict(color='green'))
    fig_price.add_scatter(x=res[res['Signal']==-1].index, y=res[res['Signal']==-1]['Close'], mode='markers', name='Sell', marker=dict(color='red'))
    fig_price.write_html('backtest_price.html')

    fig_pf = px.line(res, x=res.index, y='Cumulative Returns', title='Portfolio Value')
    fig_pf.write_html('backtest_portfolio.html')

    print('\nSaved: backtest_price.html, backtest_portfolio.html')


if __name__ == '__main__':
    run()

