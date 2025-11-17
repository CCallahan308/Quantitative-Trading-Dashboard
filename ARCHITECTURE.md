# Technical Architecture

Deep-dive into the system design and implementation details.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB DASHBOARD (Dash/Plotly)              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Control Panel          │  Results Display             │   │
│  │ • Symbol input         │  • Metric Cards             │   │
│  │ • Date range picker    │  • Price chart              │   │
│  │ • Strategy params      │  • Portfolio equity         │   │
│  │ • Model hyperparams    │  • Returns distribution     │   │
│  │ • Experiment runner    │  • Confusion matrix         │   │
│  │                        │  • ROC curve                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │    CALLBACK HANDLER LAYER (Dash)     │
        ├──────────────────────────────────────┤
        │  Input Validation                    │
        │  State Management                    │
        │  Error Handling                      │
        └──────────────────────────────────────┘
                              │
         ┌────────┬───────────┴───────────┬──────────┐
         ▼        ▼                       ▼          ▼
    ┌──────┐ ┌──────┐           ┌──────────┐   ┌──────────┐
    │Data  │ │Model │           │Backtest  │   │Experiment│
    │Fetch │ │Train │           │Engine    │   │Runner    │
    └──────┘ └──────┘           └──────────┘   └──────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────┐
    │  DATA & FEATURE ENGINEERING LAYER       │
    ├─────────────────────────────────────────┤
    │  • yfinance data fetching               │
    │  • Technical indicator computation      │
    │  • Feature normalization                │
    │  • MultiIndex flattening                │
    └─────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────┐
    │    MACHINE LEARNING LAYER (XGBoost)     │
    ├─────────────────────────────────────────┤
    │  • Walk-forward validation (65/20/15)  │
    │  • Binary classification (Up/Down)      │
    │  • Early stopping                       │
    │  • Model persistence (joblib)           │
    └─────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────┐
    │      SIGNAL GENERATION LAYER            │
    ├─────────────────────────────────────────┤
    │  Confidence Thresholds:                 │
    │    Long:  confidence > 0.75             │
    │    Short: confidence < 0.25             │
    │    Flat:  0.25 ≤ confidence ≤ 0.75     │
    └─────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────┐
    │     RISK MANAGEMENT & BACKTESTING       │
    ├─────────────────────────────────────────┤
    │  • Position sizing                      │
    │  • Stop-loss triggers                   │
    │  • Trailing stop management             │
    │  • PnL calculation                      │
    │  • Trade logging                        │
    └─────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────┐
    │    METRICS & PERSISTENCE LAYER          │
    ├─────────────────────────────────────────┤
    │  • Performance calculation              │
    │  • Run persistence (runs/<timestamp>)   │
    │  • Experiment logging (experiments/)    │
    │  • Visualization generation             │
    └─────────────────────────────────────────┘
```

## Data Flow

### Backtest Execution Pipeline

```
1. USER INPUT
   ├─ Symbol: 'SPY'
   ├─ Date Range: 2024-01-01 to 2025-11-16
   ├─ Confidence Thresholds: (0.75, 0.25)
   └─ Model Params: n_estimators=500, train_pct=65, etc.

2. DATA FETCHING
   ├─ yf.download(symbol, start, end)
   ├─ Flatten MultiIndex if needed
   └─ Validate data quality

3. FEATURE ENGINEERING
   ├─ Rolling averages (MA20, MA50)
   ├─ Momentum (RSI, MACD, ADX)
   ├─ Volatility (ATR, Bollinger Bands)
   ├─ Sentiment (TextBlob on news)
   └─ Drop NaN rows (after all features computed)

4. DATA PREPARATION
   ├─ Create target: next_return > 0? (1:0)
   ├─ Split indices:
   │  ├─ Train: 0 to 65% of data
   │  ├─ Test: 65% to 85% of data
   │  └─ Val: 85% to 100% of data
   └─ Generate X, y matrices

5. MODEL TRAINING
   ├─ Initialize XGBoost Classifier
   ├─ Fit on training set
   ├─ Early stop on test set (50 rounds default)
   └─ Compute test set metrics (AUC, confusion matrix)

6. SIGNAL GENERATION
   ├─ Predict probabilities on full dataset
   ├─ Generate signals based on confidence:
   │  ├─ confidence > 0.75 → signal = +1 (long)
   │  ├─ confidence < 0.25 → signal = -1 (short)
   │  └─ else → signal = 0 (flat)
   └─ Shift signals by 1 (trade tomorrow on today's signal)

7. BACKTESTING
   ├─ Initialize: position=None, peak_price=None
   ├─ For each day:
   │  ├─ Update prices
   │  ├─ Check entry signals (if not in trade)
   │  ├─ Check exit signals (if in trade)
   │  │  ├─ Stop-loss: pnl < -5%
   │  │  └─ Trailing stop: vol-based bounce
   │  └─ Log trade metrics
   └─ Calculate cumulative returns

8. METRICS CALCULATION
   ├─ Total Return: (final_value / initial_capital) - 1
   ├─ Annualized Return: (1 + total_return)^(252/days) - 1
   ├─ Sharpe Ratio: (annual_return / annual_volatility)
   ├─ Max Drawdown: (peak - trough) / peak
   ├─ Win Rate: winning_trades / total_trades
   └─ ROC AUC: sklearn.metrics.auc(fpr, tpr)

9. PERSISTENCE
   ├─ Save to runs/<YYYYMMDD_HHMMSS>/:
   │  ├─ model.joblib (trained XGBoost)
   │  ├─ backtest_results.csv (full OHLCV + signals + PnL)
   │  └─ metrics.json (performance metrics)
   └─ Generate visualizations

10. VISUALIZATION & OUTPUT
    ├─ Price chart with entry/exit markers
    ├─ Portfolio equity curve
    ├─ Returns histogram
    ├─ Confusion matrix heatmap
    └─ ROC curve
```

## Feature Engineering Details

### Technical Indicators

| Indicator | Calculation | Purpose |
|-----------|-------------|---------|
| **MA20/50** | `Close.rolling(20/50).mean()` | Trend identification |
| **RSI** | `100 - (100 / (1 + RS))` where RS = avg_gain/avg_loss | Momentum oscillator |
| **MACD** | `EMA(12) - EMA(26)` | Convergence/divergence |
| **Bollinger Bands** | `SMA ± (std * 2)` | Volatility bands |
| **ATR** | `TR.rolling(14).mean()` | Volatility measure |
| **ADX** | DI+ vs DI- smoothing | Trend strength |
| **Sentiment** | TextBlob polarity on news | Sentiment bias |

### Feature Normalization

```python
# Not explicitly normalized in current implementation
# XGBoost handles feature scaling internally via tree splits
# For production: consider StandardScaler or MinMaxScaler

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Model Architecture

### XGBoost Configuration

```python
XGBClassifier(
    n_estimators=500,        # Number of boosting rounds
    learning_rate=0.05,      # Shrinkage parameter
    max_depth=7,             # Tree depth limit
    subsample=0.8,           # Row sampling (80% of data per tree)
    colsample_bytree=0.8,    # Column sampling (80% of features per tree)
    gamma=0.5,               # Min loss reduction for split
    min_child_weight=1,      # Min sum of weights in child
    random_state=42,         # Reproducibility
    eval_metric='logloss'    # Binary classification metric
)
```

### Walk-Forward Validation

Why walk-forward instead of random split?

✅ **Prevents look-ahead bias**: Training only on past data
✅ **Realistic evaluation**: Simulates live trading scenario
✅ **Time-series aware**: Respects temporal order of data

```
Year 1   Year 2   Year 3   Year 4   Year 5
[TRAIN][TEST]
         [TRAIN   VAL][TEST]
                [TRAIN   VAL][TEST]
                          ... etc
```

## Backtesting Engine

### Risk Management Rules

#### Entry Signals
```python
if confidence > 0.75:
    signal = +1  # Long
elif confidence < 0.25:
    signal = -1  # Short
else:
    signal = 0   # Flat
```

#### Exit Rules
```python
if in_long_position:
    # Exit on loss > stop_loss_threshold (5% default)
    if current_return <= -0.05:
        exit_trade()
    # Exit on trailing stop breach
    if (current_price - peak_price) / peak_price >= vol_stop * volatility:
        exit_trade()

elif in_short_position:
    # Similar logic for short positions
```

#### Position Sizing
```python
# All trades are full position (100% of capital)
# Could be enhanced with dynamic sizing based on volatility
position_size = 1.0  # 100%
```

## Threading Architecture (Async Experiments)

### Problem
Long-running experiment grid searches (100+ combos) would freeze the Dash UI.

### Solution
Background thread + shared state + polling callback

```python
# Global shared state
exp_status = {
    'running': False,           # Is experiment running?
    'progress': '',             # Current progress message
    'results': None,            # Final results table (HTML)
    'results_file': None        # Path to CSV export
}
exp_lock = threading.Lock()    # Mutex for thread-safe access

# Button click: spawn worker thread
def run_experiments(n_clicks, ...):
    if exp_status['running']:
        return "Already running..."
    
    exp_status['running'] = True
    thread = threading.Thread(
        target=_run_experiments_bg,
        args=(params,),
        daemon=True
    )
    thread.start()
    return "Starting experiments..."

# Background worker
def _run_experiments_bg(params):
    for combo in parameter_combinations:
        # Train model, evaluate
        with exp_lock:
            exp_status['progress'] = f"Combo {i}/{n}: ..."
    
    with exp_lock:
        exp_status['results'] = results_table
        exp_status['running'] = False

# Polling callback (dcc.Interval every 1 second)
def poll_exp_progress(_):
    with exp_lock:
        if exp_status['results']:
            return exp_status['results']
        else:
            return exp_status['progress']
```

## Performance Characteristics

### Execution Time (SPY 2024-2025, ~500 trading days)

| Operation | Time |
|-----------|------|
| Data fetch | 2-3s |
| Feature engineering | 1-2s |
| Model training (500 trees) | 8-12s |
| Backtesting | 2-3s |
| Visualization rendering | 1-2s |
| **Total per run** | **14-22s** |

### Memory Usage

| Component | Typical |
|-----------|---------|
| Raw OHLCV data | ~10MB |
| Features (14 indicators) | ~15MB |
| XGBoost model | ~50MB |
| Backtest results | ~30MB |
| **Total** | **~105MB** |

### Scaling Behavior

```
N days of data → Linear scaling O(N)
N parameters (grid search) → Linear scaling O(N)
NxM grid search → O(NxM) backtests
```

## Persistence Strategy

### Run Directory Structure

```
runs/
├── 20251116_201355/
│   ├── model.joblib
│   │   └─ Serialized XGBoost classifier
│   │   └─ Binary format for fast loading
│   │   └─ Size: ~50MB
│   │
│   ├── backtest_results.csv
│   │   └─ Full results with columns:
│   │   │  ├─ Date, Open, High, Low, Close, Volume
│   │   │  ├─ MA20, MA50, RSI, MACD, ...
│   │   │  ├─ Signal, Confidence, PositionSize
│   │   │  ├─ PnL, ExitReason, InTrade
│   │   └─ One row per trading day
│   │
│   └── metrics.json
│       └─ JSON with all calculated metrics
│          ├─ final_value, total_return
│          ├─ sharpe, max_drawdown, win_rate
│          ├─ confusion_matrix (array)
│          └─ roc_auc (float)
```

## Error Handling

### Graceful Degradation

```python
try:
    # Compute optional metrics
    y_test_proba = model.predict_proba(X_test)
    roc_auc = auc(fpr, tpr)
except Exception:
    roc_auc = None  # Continue without metric
    
# Same for confusion matrix, sentiment, etc.
```

### Data Validation

```python
# Check data integrity
if data.empty:
    return error("No data found for symbol")
    
if len(data) < 100:
    return error("Insufficient data for features")
    
if (train_pct + val_pct) >= 100:
    return error("Invalid split percentages")
```

## Security Considerations

### Input Validation
- Symbol validation against yfinance
- Date range sanity checks
- Parameter bounds enforcement

### Data Privacy
- No external API calls with user data
- All computation local
- Optional sentiment requires internet access

### Model Security
- Serialized models should be from trusted sources
- Consider signing/checksumming model files in production

## Future Optimization Opportunities

1. **Vectorized Operations**: Use NumPy broadcasting instead of pandas .apply()
2. **Parallel Backtesting**: Use multiprocessing for grid search combos
3. **Caching**: Cache feature engineering for repeated date ranges
4. **GPU Acceleration**: XGBoost CUDA plugin for training speedup
5. **Incremental Learning**: Update model with new data vs retraining
6. **Feature Selection**: Auto-drop low-importance features
7. **Hyperparameter Optimization**: Bayesian optimization vs grid search

---

**Last Updated**: November 2025
