import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Finnhub API Configuration
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')
FINNHUB_BASE_URL = 'https://finnhub.io/api/v1'

def fetch_finnhub_data(symbol, start_date, end_date, interval='D'):
    """
    Fetch stock data from Finnhub API.
    interval: 'D' (daily), '60' (hourly), '15' (15-min), '5' (5-min)
    """
    if not FINNHUB_API_KEY:
        print("Finnhub API key not set. Skipping Finnhub.")
        return None
    
    try:
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())
        
        # Map intervals
        interval_map = {
            '1d': 'D',
            '1h': '60',
            '15m': '15',
            '5m': '5'
        }
        finnhub_interval = interval_map.get(interval, 'D')
        
        # Fetch candles
        url = f"{FINNHUB_BASE_URL}/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': finnhub_interval,
            'from': start_ts,
            'to': end_ts,
            'token': FINNHUB_API_KEY
        }
        
        print(f"Requesting Finnhub: {symbol} ({finnhub_interval})")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data_json = response.json()
        
        # Check if data is valid
        if data_json.get('s') != 'ok':
            print(f"Finnhub returned status: {data_json.get('s')}")
            return None
            
        if not data_json.get('c'):
            print("Finnhub returned empty data.")
            return None
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': data_json['o'],
            'High': data_json['h'],
            'Low': data_json['l'],
            'Close': data_json['c'],
            'Volume': data_json['v'],
        })
        
        # Convert timestamps to datetime index
        df.index = pd.to_datetime(data_json['t'], unit='s')
        df.index.name = 'Date'
        
        print(f"Successfully fetched {len(df)} rows from Finnhub for {symbol}")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Finnhub API request failed for {symbol}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error fetching Finnhub data for {symbol}: {str(e)}")
        return None

def fetch_stock_data(symbol, start_date, end_date, interval='1d', retries=3):
    """
    Fetch stock data with multiple fallback methods.
    Primary: Finnhub API
    Fallback: yfinance
    """
    # Try Finnhub first if API key is available
    if FINNHUB_API_KEY:
        print(f"Attempting to fetch {symbol} from Finnhub...")
        finnhub_data = fetch_finnhub_data(symbol, start_date, end_date, interval)
        if finnhub_data is not None and not finnhub_data.empty and len(finnhub_data) > 0:
            return finnhub_data
        print(f"Finnhub fetch failed, falling back to yfinance...")
    
    # Fallback to yfinance
    # Create a custom session with headers for each request
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    for attempt in range(retries):
        try:
            # Method 1: yfinance download with session and explicit parameters
            data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                interval=interval, 
                progress=False,
                auto_adjust=True,
                prepost=False,
                threads=True
            )
            
            if data is not None and not data.empty and len(data) > 0:
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Standardize column names
                column_mapping = {
                    'close': 'Close', 'open': 'Open', 'high': 'High', 
                    'low': 'Low', 'volume': 'Volume'
                }
                data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns}, inplace=True)
                
                print(f"Successfully fetched {len(data)} rows for {symbol} from yfinance")
                return data
        except Exception as e:
            print(f"yfinance attempt {attempt + 1} failed for {symbol}: {str(e)}")
        
        try:
            # Method 2: yfinance Ticker with history
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date, 
                end=end_date, 
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if not data.empty and len(data) > 0:
                # Standardize column names
                if 'Close' not in data.columns and 'close' in data.columns:
                    data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 
                                        'low': 'Low', 'volume': 'Volume'}, inplace=True)
                print(f"Successfully fetched {len(data)} rows for {symbol} from yfinance (Ticker method)")
                return data
        except Exception as e:
            print(f"yfinance Ticker method attempt {attempt + 1} failed for {symbol}: {str(e)}")
        
        # Wait before retry (except on last attempt)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
    
    # If all methods fail, return None
    print(f"All data fetch attempts failed for {symbol}")
    return None

def get_bars_per_year(interval):
    """
    Calculate the number of trading bars per year based on the interval.
    """
    # Assumes ~252 trading days per year, 6.5 trading hours per day
    bars_map = {
        '1d': 252,           # Daily bars
        '1h': 252 * 6.5,     # Hourly bars (~1638/year)
        '15m': 252 * 6.5 * 4,  # 15-minute bars (~6552/year)
        '5m': 252 * 6.5 * 12,  # 5-minute bars (~19656/year)
    }
    return bars_map.get(interval, 252)  # Default to daily if unknown
