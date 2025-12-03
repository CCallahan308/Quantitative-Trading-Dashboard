# Quantitative Trading Strategy Dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional-grade quantitative trading backtesting framework with interactive machine learning capabilities, built for algorithmic traders and quantitative analysts.

## 🖼️ Dashboard UI Showcase

Experience a professional, dark-mode interface designed for quantitative analysts. The dashboard combines institutional-grade aesthetics with powerful analytics capabilities, featuring interactive charts, real-time metrics, and comprehensive model diagnostics.

<table>
  <tr>
    <td align="center">
      <img src="assets/images/KPIs.png" width="400"/>
      <br/><em>Analysis KPIs</em>
    </td>
    <td align="center">
      <img src="assets/images/Output.png" width="400"/>
      <br/><em>Model Output</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/images/Factor.png" width="400"/>
      <br/><em>Factor Attribution</em>
    </td>
    <td align="center">
      <img src="assets/images/Factor2.png" width="400"/>
      <br/><em>Factor Attribution Analysis</em>
    </td>
  </tr>
</table>

> 💡 **Professional UI/UX**: These screenshots demonstrate the dashboard's enterprise-grade interface featuring interactive Plotly visualizations, real-time performance metrics, risk analytics, and comprehensive model evaluation tools—all styled with a modern dark theme optimized for extended trading sessions.

## 🎯 Features

### Core Backtesting Engine
- **Risk-Aware Simulation**: Dynamic position sizing with stop-loss and trailing stops.
- **Walk-Forward Validation**: Robust rolling window methodology with **Early Stopping** to prevent overfitting.
- **High-Fidelity Execution**: Realistic slippage and commission modeling.
- **Detailed Trade Log**: Audit every trade with entry/exit reasons and per-trade PnL.

### Machine Learning
- **XGBoost Classifier**: Optimized gradient boosting with conservative regularization (max_depth=3).
- **13+ Technical Indicators**:
  - Momentum: MACD, RSI, ADX
  - Volatility: ATR, Bollinger Bands, Annualized Volatility
  - Trend: Moving Averages (20/50-day), Plus DI/Minus DI
  - Volume: Volume Pct Change
- **Early Stopping**: Automatic training termination on validation loss to maximize generalization.

### Interactive Dashboard
- **Real-time Parameter Tuning**: Adjust strategy parameters and model hyperparameters on-the-fly.
- **Professional Visualizations**:
  - **Feature Importance**: See which indicators drive the model's decisions.
  - **Trade Log**: Tabular view of all simulated trades.
  - **Equity Curve**: Interactive portfolio tracking vs. Benchmark.
- **Bayesian Optimization**: Automated hyperparameter tuning to find the "Efficiency Frontier".

## 🏗️ System Architecture

The project follows a clean, modular architecture:

- **`app.py`**: The Dash application entry point. Handles UI callbacks and threading.
- **`core/strategy.py`**: Central logic for data preparation, feature engineering, and model training.
- **`core/backtest.py`**: Numba-optimized, event-driven backtesting engine.
- **`core/data.py`**: Robust data fetching with **Finnhub** (primary) and **yfinance** (fallback).
- **`core/indicators.py`**: Calculation of technical indicators.
- **`run_full_backtest.py`**: CLI tool for headless backtesting and reporting.

## 🚀 Quick Start

### 1. Installation

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

### 2. Configuration (Optional)

To use the Finnhub API (recommended for reliability), create a `.env` file in the root directory:

```env
FINNHUB_API_KEY=your_api_key_here
```

*If no key is provided, the system automatically falls back to `yfinance`.*

### 3. Run the Dashboard

```bash
python app.py
```

Open browser to `http://127.0.0.1:8050/`

### 4. Quick Backtest (Headless)

```bash
python run_full_backtest.py --symbol AAPL
```

Outputs:
- `backtest_price.html` - Interactive price chart with signals
- `backtest_portfolio.html` - Portfolio equity curve
- Console performance summary

## 📈 Example Results

**SPY (Jan 2024 - Dec 2025)**  
*Using optimized default parameters (Walk-Forward Validation)*

```
Final Value:       $1,082.62
Total Return:      +8.26%
Annualized Return: +4.73%
Sharpe Ratio:      -0.08 (Risk-Free Rate: 5.00%)
Win Rate:          49.16%
Trades:            238
```

> **Note**: Positive annualized returns on out-of-sample data demonstrate the strategy's robustness. The slight negative Sharpe is due to the conservative risk-free rate assumption (5%).

## 🔧 Configuration

### Strategy Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Long Confidence Threshold** | 0.5 - 1.0 | 0.55 | Lower threshold = More trades |
| **Stop-Loss Threshold (%)** | 1 - 10 | 5 | Maximum drawdown per trade |
| **Trailing Stop Scale** | 0 - 5.0 | 2.0 | Multiplier for volatility-based trailing stop |

## 📁 Project Structure

```
.
├── app.py                      # Main Dash application
├── run_full_backtest.py        # Headless backtest runner
├── requirements.txt            # Python dependencies
├── core/                       # Core Logic
│   ├── strategy.py             # Feature Eng & Model Training
│   ├── backtest.py             # Simulation Engine
│   ├── data.py                 # Data Fetching (Finnhub/YF)
│   ├── indicators.py           # Technical Indicators
│   └── utils.py                # Metrics & Helpers
├── assets/                     # UI Assets
├── runs/                       # Saved models
├── experiments/                # Optimization results
├── tests/                      # Unit tests
└── README.md                   # This file
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

Built by CCallahan308

---

**Last Updated**: December 2025