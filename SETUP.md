# Project Setup Guide

Complete guide to get the Quantitative Trading Dashboard running on your machine.

## System Requirements

- **Python**: 3.10 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB minimum (8GB recommended for large backtests)
- **Storage**: 500MB for dependencies

## Installation (Windows)

### Step 1: Clone Repository

```powershell
git clone https://github.com/CCallahan308/Quantitative-Trading-Dashboard.git
cd test
```

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
.\vbt-env\Scripts\Activate.ps1
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```powershell
python test_syntax.py
```

Expected output:
```
✓ Syntax is valid!
✓ All basic checks passed!
```

### Step 5: Start Dashboard

```powershell
python Quant_Dashboard.py
```

Navigate to `http://127.0.0.1:8050/` in your browser.

## Installation (macOS/Linux)

### Step 1: Clone Repository

```bash
git clone https://github.com/CCallahan308/test.git
cd test
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python test_syntax.py
```

### Step 5: Start Dashboard

```bash
python Quant_Dashboard.py
```

Navigate to `http://127.0.0.1:8050/` in your browser.

python Quant_Dashboard.py

### Issue: `ModuleNotFoundError: No module named 'xgboost'`

**Solution**: Ensure virtual environment is activated
CMD ["python", "Quant_Dashboard.py"]
```bash
# Windows
.\venv\Scripts\activate

black Quant_Dashboard.py
source venv/bin/activate
```

Then reinstall:
```bash
flake8 Quant_Dashboard.py --max-line-length=100
```

### Issue: Port 8050 Already in Use

**Solution**: Stop the existing process or use a different port

```python
# In Quant_Dashboard.py, change the last line:
if __name__ == '__main__':
    app.run(debug=True, port=8051)  # Use 8051 instead
```

### Issue: `yfinance` Download Fails

**Solution**: Check internet connection and yfinance API status

```python
# Test yfinance connectivity
import yfinance as yf
data = yf.download('AAPL', start='2024-01-01', end='2024-12-31')
print(data.head())
```

### Issue: Out of Memory During Backtest

**Solution**: Use shorter date ranges or lower frequency

```python
# Try 1-year lookback instead of 5 years
start_date = '2024-01-01'
end_date = '2025-01-01'
```

## Docker Setup (Optional)

### Build Docker Image

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "Quant_Dashboard.py"]
```

### Build and Run

```bash
docker build -t quant-dashboard .
docker run -p 8050:8050 quant-dashboard
```

## Development Setup

If you want to contribute or modify the code:

### Install Development Tools

```bash
pip install black flake8 pytest pytest-cov
```

### Format Code

```bash
black test\ v3.py
black run_full_backtest.py
```

### Lint Code

```bash
flake8 test\ v3.py --max-line-length=100
```

### Run Tests (if available)

```bash
pytest tests/ -v --cov=.
```

## Upgrade Dependencies

To update to the latest versions (carefully):

```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

1. Read the [README.md](README.md) for usage guide
2. Check [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
3. Review example results in `runs/` directory
4. Explore the dashboard UI and experiment with parameters

## Getting Help

- **Documentation**: See [README.md](README.md)
- **Issues**: Search [GitHub Issues](../../issues)
- **Discussions**: Open a discussion thread
- **Email**: contact@example.com

## Performance Tuning

### For Faster Backtests

```python
# Reduce date range
start_date = '2024-01-01'  # Instead of 5 years

# Reduce XGBoost trees
n_estimators = 100  # Instead of 500

# Increase train split (less validation data)
train_pct = 75  # Instead of 65
```

### For More Accurate Models

```python
# Use more trees
n_estimators = 1000

# Increase early stopping patience
early_stopping_rounds = 100

# Use longer lookback
start_date = '2015-01-01'  # 10 years
```

---

**Last Updated**: November 2025  
**Tested On**: Python 3.10-3.13, Windows 11, macOS 13+, Ubuntu 22.04
