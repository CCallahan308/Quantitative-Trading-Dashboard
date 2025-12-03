import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

def compute_risk_parity_weights(returns_df):
    try:
        volatilities = returns_df.std()
        if volatilities.sum() == 0:
            return pd.Series(1/len(returns_df), index=returns_df.columns)
        inv_vol = 1.0 / (volatilities + 1e-8)
        weights = inv_vol / inv_vol.sum()
        return weights
    except Exception as e:
        print(f"Error computing risk parity weights: {e}")
        return pd.Series(1/len(returns_df), index=returns_df.columns)

def analyze_risk_parity(data, returns_col='Returns'):
    try:
        factor_cols = ['MA20', 'MA50', 'RSI', 'Volatility', 'MACD', 'BB_Position', 'ATR', 'ADX']
        available_factors = [col for col in factor_cols if col in data.columns]
        
        if not available_factors:
            return None
        
        factor_vols = {}
        factor_returns_df = pd.DataFrame()
        
        for col in available_factors:
            col_data = data[col].dropna()
            if len(col_data) > 1:
                col_returns = col_data.pct_change().fillna(0)
                factor_returns_df[col] = col_returns
                factor_vols[col] = col_returns.std()
        
        vol_array = np.array([factor_vols.get(col, 1e-8) for col in available_factors])
        inv_vol = 1.0 / (vol_array + 1e-8)
        rp_weights = inv_vol / inv_vol.sum()
        rp_weights_series = pd.Series(rp_weights, index=available_factors)
        
        portfolio_returns = (factor_returns_df * rp_weights).sum(axis=1)
        portfolio_vol = portfolio_returns.std()
        cumulative_return = (1 + portfolio_returns).cumprod()
        factor_contributions = {col: factor_vols.get(col, 0) for col in available_factors}
        
        return {
            'weights': rp_weights_series,
            'factor_returns': factor_returns_df,
            'portfolio_returns': portfolio_returns,
            'portfolio_vol': portfolio_vol,
            'factor_contributions': factor_contributions,
            'cumulative_return': cumulative_return,
            'factor_vols': factor_vols
        }
    except Exception as e:
        print(f"Error in risk parity analysis: {e}")
        return None

def compute_factor_attribution(data, returns_col='Returns'):
    try:
        factor_cols = ['MA20', 'MA50', 'RSI', 'Volatility', 'VolumePct', 'MACD', 
                       'BB_Position', 'ATR', 'ADX', 'Sentiment']
        available_factors = [col for col in factor_cols if col in data.columns]
        
        if not available_factors or returns_col not in data.columns:
            return None
        
        X_factors = data[available_factors].fillna(0)
        y_returns = data[returns_col].fillna(0)
        
        scaler = StandardScaler()
        X_factors_scaled = scaler.fit_transform(X_factors)
        X_factors_scaled = pd.DataFrame(X_factors_scaled, columns=available_factors, index=data.index)
        
        correlations = {}
        for col in available_factors:
            corr = X_factors_scaled[col].corr(y_returns)
            correlations[col] = corr
        
        window = min(20, len(data) // 4)
        factor_impacts = {}
        
        for col in available_factors:
            rolling_corr = X_factors_scaled[col].rolling(window).corr(y_returns)
            average_impact = rolling_corr.mean()
            max_impact = rolling_corr.max()
            factor_impacts[col] = {
                'avg_impact': average_impact,
                'max_impact': max_impact,
                'correlation': correlations[col]
            }
        
        sorted_factors = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'factor_correlations': correlations,
            'factor_impacts': factor_impacts,
            'sorted_factors': sorted_factors,
            'top_3_factors': sorted_factors[:3]
        }
    except Exception as e:
        print(f"Error in factor attribution: {e}")
        return None

def compute_cross_validation(model, X, y, n_splits=5):
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores,
            'n_splits': n_splits
        }
    except Exception as e:
        print(f"Cross-validation error: {e}")
        return None

def calculate_advanced_metrics(returns, rf_rate=0.05, bars_per_year=252):
    """
    Calculate a suite of professional quantitative metrics.
    """
    try:
        metrics = {}
        
        # Cleaning
        returns = returns.fillna(0)
        if len(returns) < 2:
            return {k: 0.0 for k in ['total_return', 'cagr', 'volatility', 'sharpe', 'sortino', 'calmar', 'max_dd', 'win_rate', 'profit_factor']}

        # Basic
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (bars_per_year / len(returns)) - 1
        volatility = returns.std() * np.sqrt(bars_per_year)
        
        # Sharpe
        excess_ret = cagr - rf_rate
        sharpe = excess_ret / (volatility + 1e-8)
        
        # Sortino (Downside Deviation)
        neg_returns = returns[returns < 0]
        downside_std = neg_returns.std() * np.sqrt(bars_per_year)
        sortino = excess_ret / (downside_std + 1e-8)
        
        # Drawdown
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min() # Negative value
        
        # Calmar
        calmar = cagr / (abs(max_dd) + 1e-8)
        
        # Trade stats (Approximate based on daily returns)
        # For more accuracy we need the PnL per trade list, but this is a good approximation
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / (len(returns) + 1e-8) # This is daily win rate, not trade win rate
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        profit_factor = (wins.sum()) / (abs(losses.sum()) + 1e-8)
        
        metrics['total_return'] = total_return
        metrics['cagr'] = cagr
        metrics['volatility'] = volatility
        metrics['sharpe'] = sharpe
        metrics['sortino'] = sortino
        metrics['max_dd'] = max_dd
        metrics['calmar'] = calmar
        metrics['win_rate_daily'] = win_rate
        metrics['profit_factor_daily'] = profit_factor
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}

def calculate_trade_metrics(trades_df, capital=1000):
    """
    Calculate metrics based on the list of individual trades.
    Requires columns: 'PnL', 'ExitReason'.
    """
    if trades_df is None or trades_df.empty:
        return {
            'win_rate': 0.0, 'profit_factor': 0.0, 'avg_win_pct': 0.0, 'avg_loss_pct': 0.0,
            'trades_count': 0, 'expectancy': 0.0
        }
    
    try:
        # PnL is fractional return (e.g. 0.05 = 5%)
        wins = trades_df[trades_df['PnL'] > 0]['PnL']
        losses = trades_df[trades_df['PnL'] <= 0]['PnL']
        
        n_trades = len(trades_df)
        n_wins = len(wins)
        n_losses = len(losses)
        
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        
        avg_win = wins.mean() if n_wins > 0 else 0.0
        avg_loss = losses.mean() if n_losses > 0 else 0.0 # This is negative
        
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        if gross_loss == 0 and gross_profit == 0:
            profit_factor = 0.0
            
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'trades_count': n_trades,
            'expectancy': expectancy
        }
    except Exception as e:
        print(f"Error calculating trade metrics: {e}")
        return {}