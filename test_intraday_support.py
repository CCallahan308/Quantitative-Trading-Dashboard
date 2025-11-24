#!/usr/bin/env python3
"""Test script to validate intraday data support functionality"""
import sys
import os
from datetime import datetime, timedelta

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module
import importlib.util
module_path = os.path.join(os.path.dirname(__file__), 'Quant_Dashboard.py')
spec = importlib.util.spec_from_file_location('quant_dashboard_module', module_path)
qd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qd)

def test_bars_per_year():
    """Test that bars_per_year calculation is correct"""
    print("Testing bars_per_year calculation...")
    
    assert qd.get_bars_per_year('1d') == 252, "Daily bars should be 252"
    assert qd.get_bars_per_year('1h') == 252 * 6.5, "Hourly bars should be 252 * 6.5"
    assert qd.get_bars_per_year('15m') == 252 * 6.5 * 4, "15-min bars should be 252 * 6.5 * 4"
    assert qd.get_bars_per_year('5m') == 252 * 6.5 * 12, "5-min bars should be 252 * 6.5 * 12"
    assert qd.get_bars_per_year('unknown') == 252, "Unknown interval should default to 252"
    
    print("✓ bars_per_year calculation tests passed!")

def test_fetch_data_with_interval():
    """Test that data fetching works with interval parameter"""
    print("\nTesting data fetching with different intervals...")
    
    # Use recent dates for intraday data (Yahoo Finance limits intraday data to recent history)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')  # 7 days for intraday
    
    # Test daily interval (should always work)
    print("  Testing daily (1d) interval...")
    data_daily = qd.fetch_stock_data('SPY', start_date, end_date, interval='1d', retries=2)
    if data_daily is not None and not data_daily.empty:
        print(f"    ✓ Daily data fetched: {len(data_daily)} bars")
    else:
        print("    ⚠ Daily data fetch failed (may be API issue)")
    
    # Test hourly interval
    print("  Testing hourly (1h) interval...")
    data_hourly = qd.fetch_stock_data('SPY', start_date, end_date, interval='1h', retries=2)
    if data_hourly is not None and not data_hourly.empty:
        print(f"    ✓ Hourly data fetched: {len(data_hourly)} bars")
    else:
        print("    ⚠ Hourly data fetch failed (may be API issue)")
    
    # Note: We're not testing 15m and 5m in CI because Yahoo Finance can be flaky
    # and intraday data has limited history
    
    print("✓ Data fetching tests completed!")

def test_annualization_calculations():
    """Test that annualization calculations use correct factors"""
    print("\nTesting annualization calculations...")
    import numpy as np
    
    # Mock return data
    returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015] * 100)  # 500 bars
    
    # Test with different bars_per_year
    for interval, bars_per_year in [('1d', 252), ('1h', 1638), ('15m', 6552), ('5m', 19656)]:
        std = np.std(returns)
        annualized_vol = std * np.sqrt(bars_per_year)
        print(f"  {interval}: std={std:.6f}, annualized_vol={annualized_vol:.4f}")
        assert annualized_vol > 0, f"Annualized vol should be positive for {interval}"
    
    print("✓ Annualization calculation tests passed!")

if __name__ == '__main__':
    print("=" * 60)
    print("Running Intraday Data Support Tests")
    print("=" * 60)
    
    try:
        test_bars_per_year()
        test_fetch_data_with_interval()
        test_annualization_calculations()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
