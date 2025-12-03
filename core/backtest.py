import numpy as np
import pandas as pd
import numba
from numba import float64, boolean, int32

@numba.jit(nopython=True)
def _simulate_risk_aware_backtest_numba(
    close_prices: float64[:],
    signals: float64[:],
    volatility: float64[:],
    loss_threshold: float64,
    trail_vol_scale: float64,
    position_size_clip_min: float64,
    position_size_clip_max: float64
):
    
    n = len(close_prices)
    in_trade = np.zeros(n, dtype=numba.boolean)
    entry_price_arr = np.full(n, np.nan, dtype=numba.float64)
    exit_reason_idx = np.zeros(n, dtype=numba.int32) # 0: None, 1: stop-loss, 2: trailing-stop, 3: short-stop, 4: short-trailing
    pnl_arr = np.zeros(n, dtype=numba.float64)
    position_size_arr = np.clip(0.05 / (volatility + 1e-6), position_size_clip_min, position_size_clip_max)

    current_position = 0 # 0: None, 1: Long, -1: Short
    current_entry_price = 0.0
    current_peak_price = 0.0 # for long: max, for short: min
    current_entry_idx = -1

    # Trade Log Arrays (Dynamic resizing is hard in Numba, so we pre-allocate max possible trades = n)
    # We will return the count to slice later
    trade_entry_idx = np.zeros(n, dtype=numba.int32)
    trade_exit_idx = np.zeros(n, dtype=numba.int32)
    trade_entry_price = np.zeros(n, dtype=numba.float64)
    trade_exit_price = np.zeros(n, dtype=numba.float64)
    trade_pnl = np.zeros(n, dtype=numba.float64)
    trade_reason = np.zeros(n, dtype=numba.int32)
    trade_count = 0

    for i in range(1, n):
        current_price = close_prices[i]
        signal_val = signals[i]
        
        # If not in a trade and a signal is present
        if current_position == 0 and signal_val != 0:
            current_position = int(signal_val)
            current_entry_price = current_price
            current_peak_price = current_price
            current_entry_idx = i
            in_trade[i] = True
            entry_price_arr[i] = current_entry_price
        
        # If in a trade
        elif current_position != 0 and not np.isnan(current_entry_price):
            in_trade[i] = True
            entry_price_arr[i] = current_entry_price # Propagate entry price
            
            volatility_val = volatility[i]
            volatility_component = trail_vol_scale * volatility_val / np.sqrt(252)
            vol_stop_threshold = max(volatility_component, 0.01)

            exit_trade = False
            exit_reason = 0 # No exit
            current_return = 0.0

            if current_position == 1: # Long position
                current_return = (current_price - current_entry_price) / current_entry_price
                current_peak_price = max(current_peak_price, current_price)
                trailing_return = (current_price - current_peak_price) / current_peak_price
                
                if current_return <= -loss_threshold:
                    exit_trade = True
                    exit_reason = 1 # 'stop-loss'
                elif trailing_return <= -vol_stop_threshold:
                    exit_trade = True
                    exit_reason = 2 # 'trailing-stop'

            elif current_position == -1: # Short position
                current_return = (current_entry_price - current_price) / current_entry_price
                current_peak_price = min(current_peak_price, current_price)
                
                if current_return <= -loss_threshold:
                    exit_trade = True
                    exit_reason = 3 # 'short-stop'
                elif (current_price - current_peak_price) / current_peak_price >= vol_stop_threshold:
                    exit_trade = True
                    exit_reason = 4 # 'short-trailing'
            
            # Check for signal reversal or end of data force close
            if not exit_trade and (i == n - 1 or (signals[i] != 0 and signals[i] != current_position)):
                 exit_trade = True
                 exit_reason = 5 # 'signal-reversal/force-close'

            if exit_trade:
                pnl_arr[i] = current_return
                exit_reason_idx[i] = exit_reason
                
                # Log the trade
                trade_entry_idx[trade_count] = current_entry_idx
                trade_exit_idx[trade_count] = i
                trade_entry_price[trade_count] = current_entry_price
                trade_exit_price[trade_count] = current_price
                trade_pnl[trade_count] = current_return
                trade_reason[trade_count] = exit_reason
                trade_count += 1

                current_position = 0 # Exit trade
                current_entry_price = 0.0 # Reset
                current_peak_price = 0.0 # Reset
                current_entry_idx = -1
            else:
                pnl_arr[i] = current_return
    
    return (in_trade, entry_price_arr, exit_reason_idx, pnl_arr, position_size_arr,
            trade_entry_idx[:trade_count], trade_exit_idx[:trade_count], 
            trade_entry_price[:trade_count], trade_exit_price[:trade_count], 
            trade_pnl[:trade_count], trade_reason[:trade_count])

def simulate_risk_aware_backtest(df, loss_threshold=0.05, trail_vol_scale=0.05):
    # Prepare NumPy arrays for Numba function
    close_prices = df['Close'].to_numpy(dtype=np.float64)
    signals = df['Signal'].fillna(0).to_numpy(dtype=np.float64) # Fillna needed for numba to work correctly
    volatility = df['Volatility'].fillna(0).to_numpy(dtype=np.float64) # Fillna needed
    
    # Define constants for position_size clipping
    position_size_clip_min = 0.01
    position_size_clip_max = 0.10

    # Call the Numba-optimized function
    (in_trade_arr, entry_price_arr, exit_reason_idx_arr, pnl_arr, position_size_arr,
     t_entry_idx, t_exit_idx, t_entry_price, t_exit_price, t_pnl, t_reason) = \
        _simulate_risk_aware_backtest_numba(
            close_prices,
            signals,
            volatility,
            loss_threshold,
            trail_vol_scale,
            position_size_clip_min,
            position_size_clip_max
        )

    # Reassemble results into DataFrame
    df_copy = df.copy() # Operate on a copy
    df_copy['InTrade'] = in_trade_arr
    df_copy['EntryPrice'] = entry_price_arr
    
    # Map integer exit reasons back to strings
    exit_reason_map = {
        0: '',
        1: 'stop-loss',
        2: 'trailing-stop',
        3: 'short-stop',
        4: 'short-trailing',
        5: 'signal/close'
    }
    df_copy['ExitReason'] = np.vectorize(exit_reason_map.get)(exit_reason_idx_arr)
    
    df_copy['PnL'] = pnl_arr
    df_copy['PositionSize'] = position_size_arr

    # Create Trade Log DataFrame
    trades_df = pd.DataFrame({
        'EntryIndex': t_entry_idx,
        'ExitIndex': t_exit_idx,
        'EntryPrice': t_entry_price,
        'ExitPrice': t_exit_price,
        'PnL': t_pnl,
        'ExitReasonCode': t_reason
    })
    
    # Map exit reasons in trade log
    trades_df['ExitReason'] = trades_df['ExitReasonCode'].map(exit_reason_map)
    
    # Add dates if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        trades_df['EntryDate'] = df.index[trades_df['EntryIndex']]
        trades_df['ExitDate'] = df.index[trades_df['ExitIndex']]
    
    return {
        'data': df_copy,
        'trades': trades_df
    }
