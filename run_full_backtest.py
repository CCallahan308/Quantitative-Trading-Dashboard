#!/usr/bin/env python3
"""Headless runner to execute the full backtest and print/save results."""
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px

import importlib.util
import os

# Load the 'Quant_Dashboard.py' module dynamically
module_path = os.path.join(os.path.dirname(__file__), 'Quant_Dashboard.py')
spec = importlib.util.spec_from_file_location('quant_dashboard_module', module_path)
tv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv)


def run(symbol='SPY', start_date='2024-01-01', end_date=None, interval='1d', capital=1000, min_conf=0.75, max_conf=0.25, loss_threshold_pct=5, trail_vol_scale=0.05):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    loss_threshold = loss_threshold_pct / 100.0
    
    # Calculate bars per year based on interval
    bars_per_year = tv.get_bars_per_year(interval)

    # Fetch data
    data = tv.yf.download(symbol, start=start_date, end=end_date, interval=interval)
    if data.empty:
        print('No data returned')
        return

    # Flatten MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Feature engineering (reuse same pipeline as in the app)
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = tv.compute_rsi(data['Close'])
    data['Volatility'] = data['Close'].pct_change().rolling(20).std() * (bars_per_year ** 0.5)
    data['VolumePct'] = data['Volume'].pct_change()
    data['Sentiment'] = data.index.to_series().apply(tv.fetch_news_sentiment)

    macd_line, macd_signal, macd_hist = tv.compute_macd(data['Close'])
    data['MACD'] = macd_line
    data['MACD_Signal'] = macd_signal
    data['MACD_Histogram'] = macd_hist

    bb_upper, bb_mid, bb_lower = tv.compute_bollinger_bands(data['Close'])
    data['BB_Upper'] = bb_upper
    data['BB_Middle'] = bb_mid
    data['BB_Lower'] = bb_lower
    data['BB_Position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-6)

    data['ATR'] = tv.compute_atr(data['High'], data['Low'], data['Close'])
    adx, plus_di, minus_di = tv.compute_adx(data['High'], data['Low'], data['Close'])
    data['ADX'] = adx
    data['Plus_DI'] = plus_di
    data['Minus_DI'] = minus_di

    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Return'].shift(-1) > 0).astype(int)

    data = data.dropna().copy()

    # Features
    features = ['MA20','MA50','RSI','Volatility','VolumePct','Sentiment','MACD','MACD_Signal','MACD_Histogram','BB_Position','ATR','ADX','Plus_DI','Minus_DI']
    X = data[features]

    # Load model training from the app code (same splits)
    train_size = int(len(X) * 0.65)
    test_size = int(len(X) * 0.20)

    X_train = X[:train_size]
    y_train = data['Target'][:train_size]
    X_test = X[train_size:train_size+test_size]
    y_test = data['Target'][train_size:train_size+test_size]

    # Train model
    model = tv.xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.5,
        min_child_weight=1,
        random_state=42,
        eval_metric='logloss'
    )
    try:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
    except TypeError:
        model.fit(X_train, y_train)

    data['Confidence'] = pd.Series(model.predict_proba(X)[:,1], index=X.index)
    data['Signal'] = (data['Confidence'] > min_conf).astype(int)
    data['Signal'] = data['Signal'].where(~(data['Confidence'] < max_conf), -1)
    data['Signal'] = data['Signal'].shift(1)

    # Backtest
    res = tv.simulate_risk_aware_backtest(data, loss_threshold, trail_vol_scale)
    res['Returns'] = res['PnL'] * res['PositionSize']
    res['Cumulative Returns'] = (1 + res['Returns']).cumprod() * capital

    final_value = res['Cumulative Returns'].iloc[-1] if not res['Cumulative Returns'].empty else capital
    total_return = (final_value / capital) - 1
    annualized_return = (1 + total_return) ** (bars_per_year / len(res)) - 1 if len(res)>0 else 0
    daily_returns = res['Returns'].fillna(0)
    std_daily = daily_returns.std()
    sharpe = (annualized_return) / (std_daily * (bars_per_year ** 0.5)) if std_daily > 1e-6 else 0
    win_rate = (res['Returns'] > 0).sum() / (res['PnL'] != 0).sum() if (res['PnL'] != 0).sum() > 0 else 0
    trades = (res['PnL'] != 0).sum()

    print('--- Backtest Summary ---')
    print(f'Symbol: {symbol}  Interval: {interval}  Period: {start_date} to {end_date}')
    print(f'Final Value: ${final_value:,.2f}')
    print(f'Total Return: {total_return:.2%}  Annualized: {annualized_return:.2%}')
    print(f'Annual Volatility: {std_daily * (bars_per_year**0.5):.2%}  Sharpe: {sharpe:.2f}')
    print(f'Win rate: {win_rate:.2%}  Trades: {trades}')

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
