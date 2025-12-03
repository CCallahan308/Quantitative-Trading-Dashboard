import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
from core.strategy import Strategy
from core.indicators import fetch_news_sentiment

class TestCore(unittest.TestCase):

    def setUp(self):
        self.strategy = Strategy()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.mock_data = pd.DataFrame({
            'Open': np.random.rand(100) * 100,
            'High': np.random.rand(100) * 100,
            'Low': np.random.rand(100) * 100,
            'Close': np.random.rand(100) * 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Ensure High is highest, Low is lowest
        self.mock_data['High'] = self.mock_data[['Open', 'Close']].max(axis=1) + 1
        self.mock_data['Low'] = self.mock_data[['Open', 'Close']].min(axis=1) - 1

    def test_sentiment_determinism(self):
        date = datetime(2024, 1, 1)
        s1 = fetch_news_sentiment(date)
        s2 = fetch_news_sentiment(date)
        self.assertEqual(s1, s2)
        self.assertTrue(-1 <= s1 <= 1)

    @patch('core.strategy.fetch_stock_data')
    def test_prepare_data(self, mock_fetch):
        mock_fetch.return_value = self.mock_data.copy()
        
        data = self.strategy.prepare_data('FAKE', '2023-01-01', '2023-04-10')
        
        self.assertIsNotNone(data)
        self.assertIn('MA20', data.columns)
        self.assertIn('RSI', data.columns)
        self.assertIn('Target', data.columns)
        
        # Check features
        X, y = self.strategy.get_feature_data(data)
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], len(self.strategy.features))

    def test_walk_forward_validation(self):
        # Create a mock strategy with dummy data
        # We need enough data for the window
        X = pd.DataFrame(np.random.rand(200, 5), columns=['f1','f2','f3','f4','f5'])
        y = pd.Series(np.random.randint(0, 2, 200))
        
        # We need to mock train_model because we don't want to actually train XGBoost in unit tests (slow)
        # But Strategy.train_model returns an object with .predict_proba
        
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(50, 2) # arbitrary size, will need to match call
        
        # To properly mock, we can subclass or patch Strategy.train_model
        with patch.object(self.strategy, 'train_model') as mock_train:
            # Configure mock to return a model that returns correct shape predictions
            def side_effect(X_train, y_train, **kwargs):
                m = MagicMock()
                # predict_proba should return (n_samples, 2)
                m.predict_proba.side_effect = lambda x: np.random.rand(len(x), 2)
                return m
            
            mock_train.side_effect = side_effect
            
            train_size = 100
            probas = self.strategy.walk_forward_validation(X, y, train_size=train_size, step_pct=0.1)
            
            self.assertEqual(len(probas), 200)
            # First 50 (half of train_size 100) should be NaN (initial window)
            # Wait, logic is: walk_forward_window = max(100, int(train_size * 0.5))
            # max(100, 50) = 100.
            # So first 100 should be NaN.
            self.assertTrue(np.isnan(probas[:100]).all())
            self.assertFalse(np.isnan(probas[100:]).all())

if __name__ == '__main__':
    unittest.main()
