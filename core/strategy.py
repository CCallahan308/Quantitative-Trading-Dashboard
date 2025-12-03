import pandas as pd
import numpy as np
import xgboost as xgb
from core.data import fetch_stock_data, get_bars_per_year
from core.indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_adx,
    fetch_news_sentiment,
)

class Strategy:
    def __init__(self):
        # REMOVED 'Sentiment' to reduce noise
        self.features = [
            'MA20', 'MA50', 'RSI', 'Volatility', 'VolumePct',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Position',
            'ATR', 'ADX', 'Plus_DI', 'Minus_DI'
        ]
        self.target = 'Target'

    def prepare_data(self, symbol, start_date, end_date, interval='1d'):
        """
        Fetches data and computes all technical indicators/features.
        Returns the processed DataFrame with features and target.
        """
        bars_per_year = get_bars_per_year(interval)
        data = fetch_stock_data(symbol, start_date, end_date, interval)
        
        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Calculate Indicators
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = compute_rsi(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(bars_per_year)
        data['VolumePct'] = data['Volume'].pct_change()
        # Sentiment removed

        macd_line, macd_signal, macd_hist = compute_macd(data['Close'])
        data['MACD'] = macd_line
        data['MACD_Signal'] = macd_signal
        data['MACD_Histogram'] = macd_hist

        bb_upper, bb_mid, bb_lower = compute_bollinger_bands(data['Close'])
        data['BB_Position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-6)
        data['BB_Upper'] = bb_upper # Kept for potential visualization, though not in self.features
        data['BB_Middle'] = bb_mid
        data['BB_Lower'] = bb_lower

        data['ATR'] = compute_atr(data['High'], data['Low'], data['Close'])
        adx, plus_di, minus_di = compute_adx(data['High'], data['Low'], data['Close'])
        data['ADX'] = adx
        data['Plus_DI'] = plus_di
        data['Minus_DI'] = minus_di

        # Create Target
        data['Return'] = data['Close'].pct_change()
        data['Target'] = (data['Return'].shift(-1) > 0).astype(int)

        # Clean
        data = data.dropna().copy()
        return data

    def get_feature_data(self, data):
        """Returns X (features) and y (target) from the prepared dataframe."""
        return data[self.features], data[self.target]

    def train_model(self, X_train, y_train, X_val=None, y_val=None, n_estimators=500, early_stopping_rounds=50, params=None):
        """
        Trains the XGBoost model.
        """
        default_params = {
            'learning_rate': 0.05,
            'max_depth': 3, # Reduced from 7 to prevent overfitting
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.5,
            'min_child_weight': 1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        if params:
            default_params.update(params)

        model = xgb.XGBClassifier(n_estimators=n_estimators, **default_params)

        if early_stopping_rounds and X_val is not None and y_val is not None:
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            except TypeError:
                # Fallback for older XGBoost versions or if early_stopping_rounds causes issues
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        return model

    def walk_forward_validation(self, X, y, train_size, n_estimators=500, step_pct=0.05):
        """
        Performs rolling walk-forward validation with INTERNAL EARLY STOPPING.
        The training window is split into train (85%) and validation (15%) to prevent overfitting.
        Returns predicted probabilities for the entire dataset.
        """
        y_proba_all = np.zeros(len(X))
        walk_forward_window = max(100, int(train_size * 0.5)) # Ensure reasonable starting window
        step_size = max(1, int(len(X) * step_pct))
        
        y_proba_all[:walk_forward_window] = np.nan
        
        end_idx = walk_forward_window # Initialize end_idx

        # Loop
        for start_idx in range(walk_forward_window, len(X) - step_size, step_size):
            end_idx = min(start_idx + step_size, len(X))
            
            # Get the full training window up to this point
            X_window = X.iloc[:start_idx]
            y_window = y.iloc[:start_idx]
            
            # Split into Train/Val for Early Stopping (robustness)
            split_idx = int(len(X_window) * 0.85)
            X_wf_train, X_wf_val = X_window.iloc[:split_idx], X_window.iloc[split_idx:]
            y_wf_train, y_wf_val = y_window.iloc[:split_idx], y_window.iloc[split_idx:]
            
            X_wf_pred = X.iloc[start_idx:end_idx]
            
            try:
                # Re-train model with early stopping
                wf_model = self.train_model(
                    X_wf_train, y_wf_train,
                    X_val=X_wf_val, y_val=y_wf_val,
                    n_estimators=n_estimators,
                    early_stopping_rounds=50
                )
                y_proba_all[start_idx:end_idx] = wf_model.predict_proba(X_wf_pred)[:, 1]
            except Exception:
                y_proba_all[start_idx:end_idx] = np.nan
        
        # Final chunk if any
        if end_idx < len(X):
            try:
                X_window = X.iloc[:end_idx]
                y_window = y.iloc[:end_idx]
                split_idx = int(len(X_window) * 0.85)
                X_wf_train, X_wf_val = X_window.iloc[:split_idx], X_window.iloc[split_idx:]
                y_wf_train, y_wf_val = y_window.iloc[:split_idx], y_window.iloc[split_idx:]
                
                X_wf_pred = X.iloc[end_idx:]
                
                wf_model = self.train_model(
                    X_wf_train, y_wf_train,
                    X_val=X_wf_val, y_val=y_wf_val,
                    n_estimators=n_estimators,
                    early_stopping_rounds=50
                )
                y_proba_all[end_idx:] = wf_model.predict_proba(X_wf_pred)[:, 1]
            except Exception:
                y_proba_all[end_idx:] = np.nan
                
        return y_proba_all
