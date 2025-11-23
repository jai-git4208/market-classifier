import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class CleanEnergyDataLoader:
    """Download and prepare clean energy stock data"""
    
    def __init__(self, tickers=['ICLN', 'TAN', 'ENPH', 'FSLR']):
        self.tickers = tickers
        self.raw_data = None
        self.processed_data = None
        
    def download_data(self, period='6mo', interval='1d', max_retries=3):
        """Download latest data from Yahoo Finance with retry logic"""
        print(f"Downloading {period} of data for {self.tickers}...")
        
        for attempt in range(max_retries):
            try:
                # Try downloading all tickers at once
                data = yf.download(
                    self.tickers, 
                    period=period, 
                    interval=interval, 
                    progress=False,
                    group_by='ticker',
                    auto_adjust=True,
                    threads=True
                )
                
                if not data.empty:
                    print(f"✓ Successfully downloaded {len(data)} days of data")
                    self.raw_data = data
                    return data
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        # If all retries fail, try downloading individually
        print("\nTrying individual ticker downloads...")
        return self._download_individually(period, interval)
    
    def _download_individually(self, period='1mo', interval='1d'):
        """Download each ticker separately as fallback"""
        all_data = {}
        
        for ticker in self.tickers:
            try:
                print(f"  Downloading {ticker}...", end=" ")
                ticker_data = yf.download(
                    ticker, 
                    period=period, 
                    interval=interval, 
                    progress=False,
                    auto_adjust=True
                )
                
                if not ticker_data.empty:
                    all_data[ticker] = ticker_data
                    print(f"✓ {len(ticker_data)} days")
                else:
                    print("✗ No data")
                    
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        if not all_data:
            raise ValueError("Failed to download data for any ticker. Please check your internet connection.")
        
        # Combine into multi-index DataFrame
        combined = pd.concat(all_data, axis=1)
        combined.columns.names = ['Ticker', 'Price']
        
        self.raw_data = combined
        print(f"\n✓ Downloaded data for {len(all_data)} tickers")
        return combined
    
    def get_ticker_data(self, ticker):
        """Extract data for a specific ticker"""
        if self.raw_data is None:
            raise ValueError("No data downloaded. Call download_data() first.")
        
        try:
            # Handle multi-level columns
            if isinstance(self.raw_data.columns, pd.MultiIndex):
                df = self.raw_data[ticker].copy()
            else:
                # Handle flat columns with ticker prefix
                cols = [col for col in self.raw_data.columns if ticker in str(col)]
                df = self.raw_data[cols].copy()
                df.columns = [col.split('_')[0] if '_' in str(col) else col for col in df.columns]
            
            # Ensure standard column names
            if 'Close' not in df.columns and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            
            return df
            
        except Exception as e:
            raise ValueError(f"Could not extract data for {ticker}: {e}")
    
    def combine_all_tickers(self):
        """Combine features from all tickers into single dataframe"""
        all_features = []
        
        for ticker in self.tickers:
            try:
                ticker_df = self.get_ticker_data(ticker)
                ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
                all_features.append(ticker_df)
            except Exception as e:
                print(f"Warning: Skipping {ticker} - {e}")
                continue
        
        if not all_features:
            raise ValueError("No ticker data available to combine")
        
        combined = pd.concat(all_features, axis=1)
        combined = combined.dropna()
        return combined