# Contributing to Quantitative Trading Dashboard

Thank you for your interest in contributing! We welcome contributions that improve the quality, functionality, and documentation of this project.

## Code of Conduct

- Be respectful and professional
- Assume good intent
- Provide constructive feedback
- Report issues responsibly

## Getting Started

### Prerequisites
- Python 3.10+
- Git
- Virtual environment tool (venv)

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/quant-trading-dashboard.git
cd quant-trading-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest  # Additional dev tools
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-daily-returns-chart` ✅
- `bugfix/fix-sentiment-nans` ✅
- `docs/update-readme` ✅
- `feat-new-stuff` ❌

### 2. Make Changes

- Keep commits atomic and logical
- Write clear commit messages following conventional commits:
  ```
  feat: add support for intraday bars
  fix: resolve NaN in sentiment calculation
  docs: clarify walk-forward validation methodology
  ```

### 3. Code Style

Follow PEP 8 style guidelines:

```bash
# Format code with black
black test\ v3.py

# Check for linting issues
flake8 test\ v3.py
```

### 4. Test Your Changes

```bash
# Validate syntax
python test_syntax.py

# Manual testing
python "test v3.py"
# Navigate to http://127.0.0.1:8050 and test features
```

### 5. Commit & Push

```bash
git add .
git commit -m "feat: add feature description"
git push origin feature/your-feature-name
```

### 6. Open Pull Request

- Reference any related issues: `Closes #123`
- Provide clear description of changes
- Explain motivation and testing approach
- Include screenshots/GIFs for UI changes

## Types of Contributions

### 🐛 Bug Fixes

Submit an issue first if it's a complex bug:

1. Describe the issue clearly
2. Provide minimal reproducible example
3. Include expected vs actual behavior
4. Note Python/OS versions

### ✨ New Features

For new features, please open an issue for discussion first:

1. Describe the feature and use case
2. Explain implementation approach
3. Discuss any breaking changes
4. Get feedback before implementation

### 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings for functions
- Include examples for new parameters
- Clarify technical concepts

### 🧪 Tests

We encourage test contributions:

```python
# tests/test_indicators.py
def test_compute_rsi():
    prices = pd.Series([1, 2, 3, 4, 5] * 10)
    rsi = compute_rsi(prices)
    assert len(rsi) == len(prices)
    assert (rsi >= 0).all() and (rsi <= 100).all()
```

## Code Review Guidelines

All PRs require review. Reviewers will check:

- ✅ Code quality and style
- ✅ Tests pass and coverage maintained
- ✅ Documentation updated
- ✅ No breaking changes (or clearly noted)
- ✅ Performance implications considered

## Performance Considerations

For changes affecting backtesting:

1. Test with large datasets (5+ years of daily data)
2. Measure execution time before/after
3. Report impact in PR: `Backtest time: 12s → 10s (2s improvement)`

## Documentation Style

### Docstrings
```python
def compute_macd(prices, fast=12, slow=26, signal=9):
    """Compute MACD (Moving Average Convergence Divergence).
    
    Args:
        prices (pd.Series): Daily closing prices.
        fast (int): Fast EMA period (default 12).
        slow (int): Slow EMA period (default 26).
        signal (int): Signal EMA period (default 9).
    
    Returns:
        tuple: (macd_line, signal_line, histogram)
    
    Example:
        >>> prices = pd.Series([100, 101, 99, ...])
        >>> macd, signal, hist = compute_macd(prices)
    """
```

### Comments
```python
# Calculate daily returns with proper handling of gaps
data['Returns'] = data['Close'].pct_change().fillna(0)

# Use 1e-6 epsilon to prevent division by zero in ATR calc
atr = tr.rolling(window).mean() / (tr + 1e-6)
```

## Release Process

We follow semantic versioning: MAJOR.MINOR.PATCH

- MAJOR: Breaking changes (model format change, API rewrite)
- MINOR: New features (new indicators, experiment runner)
- PATCH: Bug fixes (NaN handling, UI tweaks)

## Issue Labels

- `bug` 🐛 Something isn't working
- `enhancement` ✨ New feature or request
- `documentation` 📚 Improvements or additions to docs
- `good first issue` 👋 Good starting point for new contributors
- `help wanted` 🤝 Need community input
- `discussion` 💭 Architecture or design discussion

## Questions or Need Help?

- Comment on relevant issue/PR
- Open a discussion for architectural questions
- Check existing documentation first

## Legal

By contributing, you agree your contributions are licensed under MIT License.

---

Thank you for improving this project! 🙏
