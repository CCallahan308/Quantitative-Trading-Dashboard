"""
This file was archived to `legacy/test.py` during repository cleanup.
The original implementation (archived) lives in `legacy/test.py`.
To run the maintained dashboard, use `Quant_Dashboard.py`.
"""

import sys

print("This legacy file has been archived to legacy/test.py. Use Quant_Dashboard.py instead.")
sys.exit(0)

        elif not position and signals[i] == -1 and not np.isnan(closes[i]):  # Short
            position = 'short'
            entry_price = closes[i]
            peak_price = entry_price
            in_trade[i] = True
            entry_prices[i] = entry_price.item()

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
            trailing_return = (peak_price - current_price) / peak_price # Corrected trailing for short

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

# Run backtest
data = data.loc[~data.index.duplicated(keep='first')]
data = simulate_risk_aware_backtest(data, loss_threshold, trail_vol_scale)

# Final returns with position size
data['Returns'] = data['PnL'] * data['PositionSize']
data['Cumulative Returns'] = (1 + data['Returns']).cumprod() * capital

# -------------------------------
# VISUALIZATION
# -------------------------------

# 1. Price + Signals
fig = px.line(data, x=data.index, y='Close', title=f'{symbol} Price & Trade Signals')
buy_signals = data[data['Signal'] == 1]
sell_signals = data[data['Signal'] == -1]

fig.add_scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal',
                marker=dict(color='green', size=8, symbol='triangle-up'))
fig.add_scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal',
                marker=dict(color='red', size=8, symbol='triangle-down'))
fig.add_scatter(x=data[data['ExitReason'].str.contains('stop')].index,
                y=data[data['ExitReason'].str.contains('stop')]['Close'],
                mode='markers', name='Stop Trigger',
                marker=dict(color='orange', size=6, symbol='circle'))

fig.update_layout(xaxis_title='Date', yaxis_title='Price', showlegend=True)
fig.show()

# 2. Cumulative Portfolio Value
fig2 = px.line(data, x=data.index, y='Cumulative Returns',
               title=f'{symbol} Backtest: Portfolio Value (Starting ${capital:,})',
               labels={'Cumulative Returns': 'Portfolio Value ($)'})
fig2.update_layout(xaxis_title='Date', yaxis_title='Value', showlegend=False)
fig2.show()

# 3. Returns Distribution
fig3 = px.histogram(data, x='Returns', title='Strategy Daily Returns Distribution',
                    labels={'Returns': 'Daily Return (%)'})
fig3.update_layout(xaxis_title='Daily Return', yaxis_title='Frequency', showlegend=False)
fig3.show()

# -------------------------------
# PERFORMANCE METRICS
# -------------------------------
final_value = data['Cumulative Returns'].iloc[-1]
total_return = (final_value / capital) - 1
annualized_return = (1 + total_return) ** (252 / len(data)) - 1
daily_returns = data['Returns'].fillna(0)
std_daily = daily_returns.std()
sharpe_ratio = (annualized_return) / (std_daily * np.sqrt(252)) if std_daily != 0 else 0
max_drawdown = (data['Cumulative Returns'].cummax() - data['Cumulative Returns']).max()
volatility_annual = std_daily * np.sqrt(252)

winning_days = (data['Returns'] > 0).sum()
total_days = (~data['Returns'].isna()).sum()
win_rate = winning_days / total_days if total_days > 0 else 0

print("="*60)
print("📊 STRATEGY PERFORMANCE SUMMARY")
print("="*60)
print(f"Symbol            : {symbol}")
print(f"Start Date        : {data.index[0].strftime('%Y-%m-%d')}")
print(f"End Date          : {data.index[-1].strftime('%Y-%m-%d')}")
print(f"Starting Capital  : ${capital:,.2f}")
print(f"Ending Value      : ${final_value:,.2f}")
print(f"Total Return      : {total_return:.2%}")
print(f"Annualized Return : {annualized_return:.2%}")
print(f"Annual Volatility : {volatility_annual:.2%}")
print(f"Sharpe Ratio      : {sharpe_ratio:.2f}")
print(f"Max Drawdown      : {max_drawdown:.2%}")
print(f"Win Rate          : {win_rate:.2%}")
print(f"Trade Count       : {data['InTrade'].sum()}")
print(f"Avg Position Size : {data['PositionSize'].mean():.2%} ± {data['PositionSize'].std():.2%}")
print("="*60)

# Optional: Print latest signal
latest = data.iloc[-1]
print(f"Latest Signal: {'Buy' if latest['Signal'] == 1 else 'Sell' if latest['Signal'] == -1 else 'Hold'} "
      f"(Confidence: {latest['Confidence']:.3f})")