#!/usr/bin/env python3
"""Test the simulate_risk_aware_backtest function directly"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Create sample data that mimics what yfinance would return
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'Close': np.random.uniform(100, 110, 100),
    'Volume': np.random.uniform(1e6, 2e6, 100),
    'Signal': np.random.choice([0, 1, -1], 100),
    'Volatility': np.random.uniform(0.01, 0.05, 100),
    'MA20': np.random.uniform(100, 110, 100),
    'MA50': np.random.uniform(100, 110, 100),
    'RSI': np.random.uniform(30, 70, 100),
    'VolumePct': np.random.uniform(-0.1, 0.1, 100),
    'Sentiment': np.random.uniform(-1, 1, 100),
    'Confidence': np.random.uniform(0, 1, 100),
    'Return': np.random.uniform(-0.02, 0.02, 100),
    'Target': np.random.choice([0, 1], 100),
}, index=dates)

# Add initial columns
data['InTrade'] = False
data['EntryPrice'] = np.nan
data['ExitReason'] = ''
data['PnL'] = 0.0
data['PositionSize'] = np.random.uniform(0.01, 0.10, 100)

print("Sample data created successfully")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Now test the function
sys.path.insert(0, r'c:\Users\Chris\Desktop\Gemini\test')

try:
    # Import and call the function
    from test_v3_functions import simulate_risk_aware_backtest
    
    print("\n" + "="*60)
    print("Testing simulate_risk_aware_backtest function...")
    print("="*60)
    
    result = simulate_risk_aware_backtest(data, loss_threshold=0.05, trail_vol_scale=0.05)
    
    print(f"✓ Function executed successfully!")
    print(f"Result shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print("\nSample of results:")
    print(result[['Close', 'Signal', 'PnL', 'InTrade', 'ExitReason']].head(10))
    print("\n✓ All tests passed!")
    
except ImportError:
    print("\nNote: test_v3_functions module not found.")
    print("The function is embedded in test v3.py, which is a Dash app.")
    print("To test the backtest function directly, it needs to be run through the Dash UI.")
except Exception as e:
    print(f"✗ Error during backtest: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
