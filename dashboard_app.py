"""
This file was moved to `legacy/dashboard_app.py` during repository cleanup.
The original implementation (archived) lives in `legacy/dashboard_app.py`.
To run the maintained dashboard, use `Quant_Dashboard.py`.
"""

import sys

print("This legacy file has been archived to legacy/dashboard_app.py. Use Quant_Dashboard.py instead.")
sys.exit(0)

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
        dcc.DatePickerRange(
            id='input-date-range',
            start_date=(datetime.now() - timedelta(days=365*2)).date(),
            end_date=datetime.now().date(),
            display_format='YYYY-MM-DD',
            className='date-picker'
        ),
        
        html.Label("Starting Capital"),
        dcc.Input(id='input-capital', type='number', value=100000, className='input-field'),
        
        html.Label("Long Signal Confidence"),
        dcc.Slider(id='slider-min-confidence', min=0.5, max=1.0, step=0.05, value=0.6, marks={i/10:str(i/10) for i in range(5, 11)}),
        
        html.Label("Short Signal Confidence"),
        dcc.Slider(id='slider-max-confidence', min=0.0, max=0.5, step=0.05, value=0.4, marks={i/10:str(i/10) for i in range(0, 6)}),
        
        html.Label("Stop-Loss Threshold (%)"),
        dcc.Slider(id='slider-loss-threshold', min=1, max=10, step=0.5, value=5, marks={i:f'{i}%' for i in range(1, 11)}),
        
        html.Label("Trailing Stop Volatility Scale"),
        dcc.Slider(id='slider-trail-vol-scale', min=0, max=0.2, step=0.01, value=0.05, marks={i/100:str(i/100) for i in range(0, 21, 5)}),
        
        html.Button('Run Backtest', id='run-button', n_clicks=0, className='run-button')
    ])

# =============================================================================
# App Layout
# =============================================================================
app.layout = html.Div(id='main-container', children=[
    html.H1(app.title),
    html.Div(className='app-content', children=[
        build_control_panel(),
        dcc.Loading(id="loading-spinner", type="default", children=html.Div(id='results-output'))
    ])
])

# =============================================================================
# Core Logic & Callbacks
# =============================================================================
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

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

def simulate_risk_aware_backtest(df, loss_threshold=0.05, trail_vol_scale=0.05):
    df = df.copy()
    df['InTrade'], df['EntryPrice'], df['ExitReason'], df['PnL'] = [False, np.nan, '', 0.0]
    position, entry_price, peak_price = None, 0.0, 0.0
    
    df['PositionSize'] = np.clip(0.05 / (df['Volatility'] + 1e-6), 0.01, 0.10)

    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # --- I'm checking for a valid 'Close' price before proceeding ---
        current_price = row['Close']
        if pd.isna(current_price):
            continue # I'll just skip this day if there's no price data
        
        if not position and row['Signal'] != 0:
            position = 'long' if row['Signal'] == 1 else 'short'
            entry_price = peak_price = current_price
            df.loc[df.index[i], ['InTrade', 'EntryPrice']] = [True, entry_price]
        
        elif position and not pd.isna(entry_price):
            # I moved this check to the top
            if pd.isna(current_price):
                continue
                
            vol_stop_threshold = max(trail_vol_scale * row['Volatility'] / np.sqrt(252), 0.01)
            
            if position == 'long':
                # --- I added a check here to prevent division by zero ---
                current_return = (current_price - entry_price) / entry_price if entry_price != 0 else 0
                peak_price = max(peak_price, current_price)
                
                # --- I added another check here ---
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
                # --- I added a check here as well ---
                current_return = (entry_price - current_price) / entry_price if entry_price != 0 else 0
                peak_price = min(peak_price, current_price)
                
                exit_trade = False
                exit_reason = ''

                if current_return <= -loss_threshold:
                    exit_trade = True
                    exit_reason = 'short-stop'
                else:
                    # Your original check here was good, but I'll make it safer
                    if peak_price > 1e-6: # I use a small number instead of 0
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
    Output('results-output', 'children'),
    Input('run-button', 'n_clicks'),
    [State('input-symbol', 'value'),
     State('input-date-range', 'start_date'),
     State('input-date-range', 'end_date'),
     State('input-capital', 'value'),
     State('slider-min-confidence', 'value'),
     State('slider-max-confidence', 'value'),
     State('slider-loss-threshold', 'value'),
     State('slider-trail-vol-scale', 'value')]
)
def run_backtest(n_clicks, symbol, start_date, end_date, capital, min_confidence_long, max_confidence_short, loss_threshold_pct, trail_vol_scale):
    if n_clicks == 0:
        return html.Div("Set parameters and click 'Run Backtest' to start.", style={'textAlign': 'center', 'marginTop': '50px'})

    try:
        loss_threshold = loss_threshold_pct / 100.0
        
        # 1. Fetch & Feature Engineering
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        if data.empty:
            return html.Div(f"No data found for symbol '{symbol}'. Check the symbol or date range.", style={'color': 'red', 'textAlign': 'center'})

        data.dropna(inplace=True)
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = compute_rsi(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        data['VolumePct'] = data['Volume'].pct_change()
        data['Sentiment'] = data.index.to_series().apply(fetch_news_sentiment)
        data['Return'] = data['Close'].pct_change()
        data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)
        data = data.dropna().copy()

        if len(data) < 50: # Need enough data for features and training
             return html.Div("Not enough data to run the backtest. Please select a wider date range.", style={'color': 'red', 'textAlign': 'center'})

        # 2. Machine Learning
        features = ['MA20', 'MA50', 'RSI', 'Volatility', 'VolumePct', 'Sentiment']
        target = 'Target'
        X, y = data[features], data[target]
        
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]

        model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss')
        model_xgb.fit(X_train, y_train)

        y_proba = model_xgb.predict_proba(X)[:, 1]
        data['Confidence'] = pd.Series(y_proba, index=data.index)
        data['Signal'] = np.where(data['Confidence'] > min_confidence_long, 1, np.where(data['Confidence'] < max_confidence_short, -1, 0))
        data['Signal'] = data['Signal'].shift(1)

        # 3. Backtest
        data = simulate_risk_aware_backtest(data, loss_threshold, trail_vol_scale)
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
```
============================================================
📊 STRATEGY PERFORMANCE SUMMARY
============================================================
Symbol            : {symbol.upper()}
Start Date        : {data.index[0].strftime('%Y-%m-%d')}
End Date          : {data.index[-1].strftime('%Y-%m-%d')}
------------------------------------------------------------
Starting Capital  : ${capital:,.2f}
Ending Value      : ${final_value:,.2f}
Total Return      : {total_return:.2%}
Annualized Return : {annualized_return:.2%}
------------------------------------------------------------
Annual Volatility : {volatility_annual:.2%}
Sharpe Ratio      : {sharpe_ratio:.2f}
Max Drawdown      : {max_drawdown_pct:.2%}
Win Rate (Trades) : {win_rate:.2%}
------------------------------------------------------------
Trade Count       : {(data['PnL'] != 0).sum()}
Avg Position Size : {data[data['PnL'] != 0]['PositionSize'].mean():.2%}
```
"""
        latest = data.iloc[-1]
        latest_signal_text = f"**Latest Signal**: {'Buy' if latest['Signal'] == 1 else 'Sell' if latest['Signal'] == -1 else 'Hold'} (Confidence: {latest['Confidence']:.3f})"

        # 5. Visualizations
        fig_price = px.line(data, x=data.index, y='Close', title=f'{symbol.upper()} Price & Trade Signals')
        fig_price.add_scatter(x=data[data['Signal'] == 1].index, y=data[data['Signal'] == 1]['Close'], mode='markers', name='Buy Signal', marker=dict(color='limegreen', size=9, symbol='triangle-up'))
        fig_price.add_scatter(x=data[data['Signal'] == -1].index, y=data[data['Signal'] == -1]['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=9, symbol='triangle-down'))
        fig_price.add_scatter(x=data[data['ExitReason'].str.contains('stop', na=False)].index, y=data[data['ExitReason'].str.contains('stop', na=False)]['Close'], mode='markers', name='Stop Trigger', marker=dict(color='orange', size=7, symbol='x'))

        fig_portfolio = px.line(data, x=data.index, y='Cumulative Returns', title=f'Portfolio Value (Started with ${capital:,})', labels={'Cumulative Returns': 'Portfolio Value ($)'})
        
        fig_returns = px.histogram(data, x='Returns', nbins=100, title='Strategy Daily Returns Distribution', labels={'Returns': 'Daily Return'})

        return html.Div([
            html.Div(className='summary-container', children=[
                dcc.Markdown(summary_text, className='markdown'),
                html.H5(latest_signal_text, style={'textAlign': 'center', 'fontWeight': 'bold'})
            ]),
            dcc.Graph(figure=fig_price),
            dcc.Graph(figure=fig_portfolio),
            dcc.Graph(figure=fig_returns)
        ])
    except Exception as e:
        return html.Div([
            html.H4("An error occurred:", style={'color': 'red'}),
            html.Pre(f"{e}", style={'border': '1px solid #ddd', 'padding': '10px'})
        ])

if __name__ == '__main__':
    app.run(debug=True)
