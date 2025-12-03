# Release Notes

## Version 2.0 - December 2025

### 🎉 Major Release: Unified Core Architecture, Finnhub Integration, and Advanced Walk-Forward Optimization

This is the initial v2.0 release of the Quantitative Trading Strategy Dashboard, featuring a complete professional-grade quantitative trading backtesting framework.

### ✨ Key Features

#### Core Backtesting Engine
- **Risk-Aware Simulation**: Dynamic position sizing with stop-loss and trailing stops
- **Walk-Forward Validation**: Robust rolling window methodology with Early Stopping to prevent overfitting
- **High-Fidelity Execution**: Realistic slippage and commission modeling
- **Detailed Trade Log**: Audit every trade with entry/exit reasons and per-trade PnL

#### Machine Learning
- **XGBoost Classifier**: Optimized gradient boosting with conservative regularization (max_depth=3)
- **13+ Technical Indicators**:
  - Momentum: MACD, RSI, ADX
  - Volatility: ATR, Bollinger Bands, Annualized Volatility
  - Trend: Moving Averages (20/50-day), Plus DI/Minus DI
  - Volume: Volume Pct Change
- **Early Stopping**: Automatic training termination on validation loss to maximize generalization

#### Interactive Dashboard
- **Real-time Parameter Tuning**: Adjust strategy parameters and model hyperparameters on-the-fly
- **Professional Visualizations**:
  - Feature Importance: See which indicators drive the model's decisions
  - Trade Log: Tabular view of all simulated trades
  - Equity Curve: Interactive portfolio tracking vs. Benchmark
- **Bayesian Optimization**: Automated hyperparameter tuning to find the "Efficiency Frontier"

#### Data Integration
- **Finnhub API**: Primary data source with automatic fallback to yfinance
- **Robust Error Handling**: Graceful degradation when API limits are reached
- **News Sentiment**: Integrated sentiment analysis from financial news

### 📦 What's Included

#### Core Modules
- `app.py`: Main Dash application with interactive UI
- `core/strategy.py`: Feature engineering and model training
- `core/backtest.py`: Numba-optimized event-driven backtesting engine
- `core/data.py`: Robust data fetching with dual data sources
- `core/indicators.py`: Technical indicator calculations
- `core/utils.py`: Performance metrics and utilities
- `run_full_backtest.py`: CLI tool for headless backtesting

#### Documentation
- `README.md`: Comprehensive project overview and quick start guide
- `ARCHITECTURE.md`: Detailed system architecture documentation
- `SETUP.md`: Installation and configuration guide
- `CONTRIBUTING.md`: Contribution guidelines
- `EXPERIMENTS_GUIDE.md`: Guide for running experiments and optimizations
- `QUICK_REFERENCE.md`: Quick reference for common tasks

#### Testing
- `tests/test_core.py`: Unit tests for core functionality

### 🚀 Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) Add Finnhub API key to `.env` file
4. Run the dashboard: `python app.py`
5. Open browser to `http://127.0.0.1:8050/`

### 📊 Example Performance

**SPY (Jan 2024 - Dec 2025)** - Using optimized default parameters:
- Final Value: $1,082.62
- Total Return: +8.26%
- Annualized Return: +4.73%
- Sharpe Ratio: -0.08 (Risk-Free Rate: 5.00%)
- Win Rate: 49.16%
- Trades: 238

### 🔧 Technical Details

- **Python Version**: 3.10+
- **Key Dependencies**: 
  - Dash for interactive UI
  - XGBoost for machine learning
  - Numba for performance optimization
  - Plotly for visualizations
  - scikit-optimize for Bayesian optimization

### 📝 License

MIT License - see LICENSE file for details

### 👤 Author

Built by CCallahan308

---

**Release Date**: December 2025  
**Commit**: 2bc2896c3fd18be16ce5d0fe7240e6926d437c97
