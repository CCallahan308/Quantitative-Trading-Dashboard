import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from textblob import TextBlob
import optuna
import ta

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=['/assets/style.css'])
app.title = "VBT-Backtester"

# App layout
app.layout = html.Div([
    html.H1("VBT-Backtester"),
    html.Div([
        # Control Panel
        html.Div([
            html.H3("Parameters"),
            html.Div([
                html.Label("Symbol"),
                dcc.Input(id='symbol', value='SPY', type='text'),
            ], className='control'),
            html.Div([
                html.Label("Start Date"),
                dcc.DatePickerSingle(
                    id='start-date',
                    min_date_allowed=datetime(2010, 1, 1),
                    max_date_allowed=datetime.now(),
                    initial_visible_month=datetime.now() - timedelta(days=365*2),
                    date=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
                ),
            ], className='control'),
            html.Div([
                html.Label("End Date"),
                dcc.DatePickerSingle(
                    id='end-date',
                    min_date_allowed=datetime(2010, 1, 1),
                    max_date_allowed=datetime.now(),
                    initial_visible_month=datetime.now(),
                    date=datetime.now().strftime('%Y-%m-%d')
                ),
            ], className='control'),
            html.Div([
                html.Label("Starting Capital"),
                dcc.Input(id='capital', value=100000, type='number'),
            ], className='control'),
            html.Div([
                html.Label("Loss Threshold"),
                dcc.Input(id='loss-threshold', value=0.05, type='number'),
            ], className='control'),
            html.Div([
                html.Label("Trailing Stop Scale"),
                dcc.Input(id='trail-vol-scale', value=0.05, type='number'),
            ], className='control'),
            html.Div([
                html.Label("Technical Indicators"),
                dcc.Checklist(
                    id='technical-indicators',
                    options=[
                        {'label': 'Bollinger Bands', 'value': 'bb'},
                        {'label': 'MACD', 'value': 'macd'},
                        {'label': 'Ichimoku Cloud', 'value': 'ichimoku'}
                    ],
                    value=['bb', 'macd']
                ),
            ], className='control'),
            html.Div([
                html.Label("Optimize Hyperparameters"),
                dcc.Checklist(
                    id='optimize-hyperparameters',
                    options=[{'label': 'Enable', 'value': 'True'}],
                    value=[]
                ),
            ], className='control'),
            html.Button('Run Backtest', id='run-backtest', n_clicks=0),
        ], id='control-panel'),

        # Main Content
        html.Div([
            # Charts
            html.Div(id='charts-container'),
            # Performance Metrics
            html.Div(id='performance-metrics'),
        ])
    ], id='main-container')
])

@app.callback(
    [Output('charts-container', 'children'),
     Output('performance-metrics', 'children')],
    [Input('run-backtest', 'n_clicks')],
    [State('symbol', 'value'),
     State('start-date', 'date'),
     State('end-date', 'date'),
     State('capital', 'value'),
     State('loss-threshold', 'value'),
     State('trail-vol-scale', 'value'),
     State('technical-indicators', 'value'),
     State('optimize-hyperparameters', 'value')]
)
def run_backtest(n_clicks, symbol, start_date, end_date, capital, loss_threshold, trail_vol_scale, technical_indicators, optimize_hyperparameters):
    if n_clicks == 0:
        return [], ""

    try:
        # -------------------------------
        # FETCH DATA
        # -------------------------------
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        
        # -------------------------------
        # FEATURE ENGINEERING
        # -------------------------------
        features = []
        if 'bb' in technical_indicators:
            indicator_bb = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
            data['bb_bbm'] = indicator_bb.bollinger_mavg()
            data['bb_bbh'] = indicator_bb.bollinger_hband()
            data['bb_bbl'] = indicator_bb.bollinger_lband()
            features.extend(['bb_bbm', 'bb_bbh', 'bb_bbl'])
            
        if 'macd' in technical_indicators:
            indicator_macd = ta.trend.MACD(close=data["Close"], window_slow=26, window_fast=12, window_sign=9)
            data['macd'] = indicator_macd.macd()
            data['macd_signal'] = indicator_macd.macd_signal()
            features.extend(['macd', 'macd_signal'])

        if 'ichimoku' in technical_indicators:
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

        def fetch_news_sentiment(date: datetime):
            try:
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
        features.append('Sentiment')
        
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

        if 'True' in optimize_hyperparameters:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                }
                model = xgb.XGBClassifier(**params, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                sharpe_ratio = y_pred.mean() / y_pred.std() if y_pred.std() != 0 else 0
                return -sharpe_ratio

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100)
            best_params = study.best_params
            model_xgb = xgb.XGBClassifier(**best_params, random_state=42)
        else:
            best_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
            model_xgb = xgb.XGBClassifier(**best_params, random_state=42)
        
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
        data = simulate_risk_aware_backtest(data, loss_threshold, trail_vol_scale)

        data['Returns'] = data['PnL'] * data['PositionSize']
        data['Cumulative Returns'] = (1 + data['Returns']).cumprod() * capital

        # -------------------------------
        # VISUALIZATION & METRICS
        # -------------------------------
        template = 'plotly_dark'
        
        charts = [
            dcc.Graph(
                figure=px.line(data, x=data.index, y='Close', title=f'{symbol} Price & Trade Signals', template=template)
                .add_scatter(x=data[data['Signal'] == 1].index, y=data[data['Signal'] == 1]['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='triangle-up'))
                .add_scatter(x=data[data['Signal'] == -1].index, y=data[data['Signal'] == -1]['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='triangle-down'))
                .add_scatter(x=data[data['ExitReason'].str.contains('stop')].index, y=data[data['ExitReason'].str.contains('stop')]['Close'], mode='markers', name='Stop Trigger', marker=dict(color='orange', size=6, symbol='circle'))
                .update_layout(xaxis_title='Date', yaxis_title='Price', showlegend=True)
            ),
            dcc.Graph(
                figure=px.line(data, x=data.index, y='Cumulative Returns', title=f'{symbol} Backtest: Portfolio Value', template=template)
                .update_layout(xaxis_title='Date', yaxis_title='Value')
            ),
            dcc.Graph(
                figure=px.histogram(data, x='Returns', title='Strategy Daily Returns Distribution', template=template)
                .update_layout(xaxis_title='Daily Return', yaxis_title='Frequency')
            )
        ]

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
        latest = data.iloc[-1]
        latest_signal = 'Buy' if latest['Signal'] == 1 else 'Sell' if latest['Signal'] == -1 else 'Hold'
        latest_confidence = latest['Confidence']

        metrics = html.Div([
            html.H3("Performance Metrics"),
            html.Table([
                html.Tr([html.Td("Symbol"), html.Td(symbol)]),
                html.Tr([html.Td("Start Date"), html.Td(data.index[0].strftime('%Y-%m-%d'))]),
                html.Tr([html.Td("End Date"), html.Td(data.index[-1].strftime('%Y-%m-%d'))]),
                html.Tr([html.Td("Starting Capital"), html.Td(f"${capital:,.2f}")]),
                html.Tr([html.Td("Ending Value"), html.Td(f"${final_value:,.2f}")]),
                html.Tr([html.Td("Total Return"), html.Td(f"{total_return:.2%}")]),
                html.Tr([html.Td("Annualized Return"), html.Td(f"{annualized_return:.2%}")]),
                html.Tr([html.Td("Annual Volatility"), html.Td(f"{volatility_annual:.2%}")]),
                html.Tr([html.Td("Sharpe Ratio"), html.Td(f"{sharpe_ratio:.2f}")]),
                html.Tr([html.Td("Max Drawdown"), html.Td(f"{max_drawdown:.2%}")]),
                html.Tr([html.Td("Win Rate"), html.Td(f"{win_rate:.2%}")]),
                html.Tr([html.Td("Trade Count"), html.Td(data['InTrade'].sum())]),
                html.Tr([html.Td("Avg Position Size"), html.Td(f"{data['PositionSize'].mean():.2%} ± {data['PositionSize'].std():.2%}")]),
                html.Tr([html.Td("Latest Signal"), html.Td(f"{latest_signal} (Confidence: {latest_confidence:.3f})")]),
            ])
        ])
        
        if 'True' in optimize_hyperparameters:
            param_div = html.Div([
                html.H3("Best Parameters"),
                html.Table([
                    html.Tr([html.Td(key), html.Td(str(value))]) for key, value in best_params.items()
                ])
            ])
            metrics.children.append(param_div)

        return charts, metrics
    except Exception as e:
        return [], f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
