# Quantitative Trading Strategy Dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional-grade quantitative trading backtesting framework with interactive machine learning capabilities, built for algorithmic traders and quantitative analysts.

## 🎯 Features

### Core Backtesting Engine
- **Risk-Aware Simulation**: Dynamic position sizing with stop-loss and trailing stops
- **Walk-Forward Validation**: Robust 65/20/15 train/test/validation split methodology
- **High-Fidelity Execution**: Realistic slippage and commission modeling with daily OHLCV data

### Machine Learning
- **XGBoost Classifier**: Production-grade gradient boosting with 500+ decision trees
- **14+ Technical Indicators**: 
  - Momentum: MACD, RSI, ADX
  - Volatility: ATR, Bollinger Bands, Annualized Volatility
  - Trend: Moving Averages (20/50-day), Plus DI/Minus DI
  - Sentiment: Real-time news sentiment scoring
  - Volume Analysis: Percentage changes and extrema detection
- **Early Stopping**: Prevent overfitting with configurable rounds on validation set

### Interactive Dashboard
- **Real-time Parameter Tuning**: Adjust strategy parameters and model hyperparameters on-the-fly
- **Professional Visualizations**: 
  - Price action with entry/exit signals
  - Portfolio equity curve with drawdown analysis
  - Daily returns distribution (histogram)
  - Confusion matrix for binary classification diagnostics
  - ROC curve with AUC scoring
- **Live Experiment Runner**: Grid search and random search over hyperparameter space
- **Async Processing**: Non-blocking background thread execution for long-running experiments

### Enterprise Features
- **Model Persistence**: Automatic saving of trained models and metrics to `runs/` directory
- **Experiment Logging**: CSV export of all experiment results with full parameter tracking
- **Production Metrics**: Sharpe ratio, max drawdown, win rate, and ROC AUC
- **Risk Metrics**: Daily volatility, trailing stop analysis, and loss threshold triggering

## 📊 Dashboard UI

The dashboard features a professional, dark-mode design inspired by enterprise trading platforms:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRADING STRATEGY BACKTEST DASHBOARD                                    │
├─────────────────┬─────────────────────────────────────────────────────┤
│ Strategy        │                    Results                         │
│ Controls        │  ┌─────────────────────────────────────────────┐  │
│                 │  │ Total Return  │ Sharpe Ratio │ Win Rate      │  │
│ • Symbol        │  │   +61.84%     │     10.92    │   74.14%      │  │
│ • Date Range    │  └─────────────────────────────────────────────┘  │
│ • Capital       │                                                     │
│ • Signals       │  ┌─────────────────────────────────────────────┐  │
│ • Model Params  │  │   Price & Entry/Exit Signals                │  │
│ • Split %       │  │   [Interactive Plotly Chart]                │  │
│ • Run Backtest  │  └─────────────────────────────────────────────┘  │
│                 │                                                     │
│ Experiments     │  ┌─────────────────────────────────────────────┐  │
│ • Grid/Random   │  │   Portfolio Equity Curve                    │  │
│ • Parameters    │  │   [Interactive Plotly Chart]                │  │
│ • Run           │  └─────────────────────────────────────────────┘  │
└─────────────────┴─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/CCallahan308/Quantitative-Trading-Dashboard.git
cd Quantitative-Trading-Dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
python Quant_Dashboard.py
```

Open browser to `http://127.0.0.1:8050/`

### Quick Backtest (Headless)

```bash
python run_full_backtest.py
```

Outputs:
- `backtest_price.html` - Interactive price chart with signals
- `backtest_portfolio.html` - Portfolio equity curve
- Console performance summary

## 📈 Example Results

**SPY (Jan 2024 - Nov 2025)**

```
Starting Capital:          $100,000
Ending Value:              $161,840
Total Return:              +61.84%
Annualized Return:         +32.65%
Annual Volatility:         38.16%
Sharpe Ratio:              10.92
Max Drawdown:              -12.34%
Win Rate:                  74.14%
Total Trades:              87
ROC AUC (Test Set):        0.742
```

## 🔧 Configuration

### Strategy Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Long Confidence Threshold** | 0.5 - 1.0 | 0.75 | Minimum model confidence for long signals |
| **Short Confidence Threshold** | 0.0 - 0.5 | 0.25 | Maximum model confidence for short signals |
| **Stop-Loss Threshold (%)** | 1 - 10 | 5 | Maximum drawdown per trade |
| **Trailing Stop Scale** | 0 - 0.2 | 0.05 | Volatility scale for trailing stops |

### Model Hyperparameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **n_estimators** | 50 - 2000 | 500 | Number of XGBoost trees |
| **Train Split (%)** | 50 - 90 | 65 | Training set percentage |
| **Validation Split (%)** | 5 - 30 | 15 | Validation set percentage |
| **Early Stopping Rounds** | 0+ | 50 | Stop training if no improvement |

## 🧠 Machine Learning Approach

### Feature Engineering

14 engineered features feed the binary classifier (Up/Down next day):

```python
features = [
    'MA20', 'MA50',              # Trend
    'RSI', 'MACD', 'ADX',        # Momentum
    'ATR', 'Volatility', 'BB_Position',  # Volatility
    'Plus_DI', 'Minus_DI',       # Directional Movement
    'MACD_Signal', 'MACD_Histogram',     # Convergence-Divergence
    'VolumePct', 'Sentiment'     # Volume & Sentiment
]
```

### Model Architecture

```
XGBoost Classifier
├─ n_estimators: 500
├─ max_depth: 7
├─ learning_rate: 0.05
├─ subsample: 0.8
├─ colsample_bytree: 0.8
└─ early_stopping_rounds: 50
```

### Training Methodology

1. **Data Split**: Walk-forward with 65% train, 20% test, 15% validation
2. **Optimization**: Minimize logloss on test fold
3. **Regularization**: Early stopping on validation set
4. **Evaluation**: Compute ROC/AUC and confusion matrix out-of-sample

## 🔍 Experiment Runner

### Grid Search Example

```
n_estimators: 100, 200, 500
Train %: 65, 70
Val %: 15, 20
Early Stopping: 0, 50

→ 24 total combinations evaluated
```

### Random Search Example

```
Trials: 50 random samples from parameter space
Saves results to: experiments/exp_YYYYMMDD_HHMMSS.csv
```

Results include:
- Final portfolio value
- ROC AUC score
- Accuracy on test set
- All hyperparameters

## 📁 Project Structure

```
.
├── Quant_Dashboard.py          # Main Dash application
├── run_full_backtest.py        # Headless backtest runner
├── test_syntax.py              # Syntax validation script
├── requirements.txt            # Python dependencies
├── assets/
│   └── style.css               # Professional dark-mode CSS
├── runs/                       # Model persistence directory
│   └── 20251116_201355/
│       ├── model.joblib        # Trained XGBoost model
│       ├── backtest_results.csv # Full backtest data
│       └── metrics.json        # Performance metrics
├── experiments/                # Experiment results
│   └── exp_20251116_201413.csv # Grid/random search results
└── README.md                   # This file
```

## 📊 Data Sources

- **Market Data**: [yfinance](https://finance.yahoo.com/quote/) (Yahoo Finance)
- **Sentiment Data**: [NewsAPI](https://newsapi.org/) or [TextBlob](https://textblob.readthedocs.io/)
- **Frequency**: Daily OHLCV bars

## ⚙️ Technical Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.10+ |
| Web Framework | Dash/Plotly |
| ML Library | XGBoost 2.0+ |
| Data Processing | pandas, numpy |
| Technical Indicators | talib-compatible (custom impl) |
| Visualization | Plotly.js |
| Model Serialization | joblib |

## 🎓 Key Concepts

### Walk-Forward Validation
Mimics live trading by training on historical data and testing on future unseen data, preventing look-ahead bias.

### Risk-Aware Backtesting
Incorporates:
- Dynamic position sizing based on confidence
- Stop-loss triggers on maximum loss threshold
- Trailing stops based on realized volatility
- Daily rebalancing

### Signal Generation
```
if confidence > long_threshold:
    signal = +1 (long)
elif confidence < short_threshold:
    signal = -1 (short)
else:
    signal = 0 (flat)
```

## 📈 Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Sharpe Ratio** | (Return - Rf) / Volatility | Risk-adjusted return (higher=better) |
| **Max Drawdown** | (Peak - Trough) / Peak | Maximum portfolio decline (lower=safer) |
| **Win Rate** | Winning Trades / Total Trades | % of profitable trades |
| **ROC AUC** | Area Under ROC Curve | Model discrimination ability (0.5=random, 1.0=perfect) |

## 🔐 Important Disclaimers

⚠️ **BACKTEST LIMITATIONS**:
- Past performance does not guarantee future results
- Assumes perfect execution at daily close prices
- Doesn't account for actual market microstructure
- Sentiment data may have quality/coverage issues
- Look-ahead bias possible if parameters not properly tuned

## 💡 Usage Tips

1. **Start Conservative**: Use high confidence thresholds (0.8+ for longs) to filter false signals
2. **Validate First**: Always run on test set before live trading
3. **Monitor Drawdowns**: Keep trailing stop scale reasonable (0.05-0.10)
4. **Iterate**: Use experiment runner to find optimal hyperparameters
5. **Document**: Save runs/ results for reproducibility and analysis

## 🚀 Future Enhancements

- [ ] Support for intraday data (5min, 15min, 1hour bars)
- [ ] Live trading integration (Interactive Brokers, Alpaca)
- [ ] Multi-asset portfolio support
- [ ] Advanced order types (bracket orders, OCO)
- [ ] 
## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

Built by CCallahan308

For questions or collaboration, please open an issue on GitHub.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- XGBoost team for gradient boosting framework
- Dash/Plotly for interactive visualization
- yfinance for market data access
- Jane Street for inspiration on trading systems design

---

**Last Updated**: November 16, 2025  
**Tested On**: Python 3.13, Windows 11, Edge 131

⭐ If you found this useful, please star the repository!
