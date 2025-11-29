import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class CleanEnergyDataLoader:
    """Download and prepare stock data for multiple categories (SDG + Non-SDG)"""
    
    # Predefined ticker categories
    TICKER_CATEGORIES = {
        'SDG_CLEAN_ENERGY': {
            'tickers': ['ICLN', 'TAN', 'ENPH', 'FSLR', 'RUN', 'SEDG'],
            'description': 'Clean Energy (SDG #7)',
            'sdg_aligned': True
        },
        'SDG_HEALTH': {
            'tickers': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR'],
            'description': 'Healthcare (SDG #3)',
            'sdg_aligned': True
        },
        'SDG_CLIMATE': {
            'tickers': ['TSLA', 'NEE', 'BEP', 'ENPH', 'RUN'],
            'description': 'Climate Action (SDG #13)',
            'sdg_aligned': True
        },
        'SDG_INNOVATION': {
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD'],
            'description': 'Industry & Innovation (SDG #9)',
            'sdg_aligned': True
        },
        'TECH': {
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            'description': 'Technology',
            'sdg_aligned': False
        },
        'FINANCE': {
            'tickers': ['JPM', 'BAC', 'GS', 'MS'],
            'description': 'Financial Services',
            'sdg_aligned': False
        },
        'CONSUMER': {
            'tickers': ['AMZN', 'WMT', 'HD', 'NKE'],
            'description': 'Consumer & Retail',
            'sdg_aligned': False
        },
        'ENERGY_TRADITIONAL': {
            'tickers': ['XOM', 'CVX', 'COP', 'SLB'],
            'description': 'Traditional Energy',
            'sdg_aligned': False
        }
    }
    
    def __init__(self, category='SDG_CLEAN_ENERGY', custom_tickers=None):
        """
        Initialize with category or custom tickers
        
        Args:
            category: One of the predefined categories or 'CUSTOM'
            custom_tickers: List of custom ticker symbols (if category='CUSTOM')
        """
        if custom_tickers:
            self.tickers = custom_tickers
            self.category = 'CUSTOM'
            self.description = 'Custom Selection'
            self.sdg_aligned = False
        elif category in self.TICKER_CATEGORIES:
            cat_info = self.TICKER_CATEGORIES[category]
            self.tickers = cat_info['tickers']
            self.category = category
            self.description = cat_info['description']
            self.sdg_aligned = cat_info['sdg_aligned']
        else:
            raise ValueError(f"Unknown category: {category}. Choose from {list(self.TICKER_CATEGORIES.keys())} or use custom_tickers")
        
        self.raw_data = None
        self.processed_data = None
        
        print(f"Initialized: {self.description}")
        print(f"Tickers: {self.tickers}")
        print(f"SDG Aligned: {'Yes' if self.sdg_aligned else 'No'}")
    
    @classmethod
    def list_categories(cls):
        """Print all available categories"""
        print("\n" + "="*70)
        print("AVAILABLE TICKER CATEGORIES")
        print("="*70)
        for cat, info in cls.TICKER_CATEGORIES.items():
            sdg = "✓ SDG" if info['sdg_aligned'] else "  ---"
            print(f"{sdg} | {cat:20s} | {info['description']}")
            print(f"      Tickers: {', '.join(info['tickers'])}")
        print("="*70 + "\n")
    
    def download_data(self, period='2y', interval='1d', max_retries=3):
        """Download latest data from Yahoo Finance with retry logic"""
        print(f"\nDownloading {period} of data for {self.category}...")
        
        for attempt in range(max_retries):
            try:
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
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        print("\nTrying individual ticker downloads...")
        return self._download_individually(period, interval)
    
    def _download_individually(self, period='5y', interval='1d'):
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
                    
                time.sleep(0.5)
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        if not all_data:
            raise ValueError("Failed to download data for any ticker. Check internet connection.")
        
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
            if isinstance(self.raw_data.columns, pd.MultiIndex):
                df = self.raw_data[ticker].copy()
            else:
                cols = [col for col in self.raw_data.columns if ticker in str(col)]
                df = self.raw_data[cols].copy()
                df.columns = [col.split('_')[0] if '_' in str(col) else col for col in df.columns]
            
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
    
    def get_sdg_info(self):
        """Get SDG alignment information"""
        if self.sdg_aligned:
            sdg_mapping = {
                'SDG_CLEAN_ENERGY': {
                    'sdg_number': 7,
                    'sdg_name': 'Affordable and Clean Energy',
                    'impact': 'Supports renewable energy investment and clean energy transition. Enables better capital allocation to sustainable energy projects.',
                    'targets': ['7.2', '7.3', '7.a', '7.b']
                },
                'SDG_HEALTH': {
                    'sdg_number': 3,
                    'sdg_name': 'Good Health and Well-being',
                    'impact': 'Promotes healthcare access and pharmaceutical innovation. Helps investors identify healthcare opportunities.',
                    'targets': ['3.8', '3.b', '3.d']
                },
                'SDG_CLIMATE': {
                    'sdg_number': 13,
                    'sdg_name': 'Climate Action',
                    'impact': 'Supports climate-friendly investments and carbon reduction initiatives. Aligns with Paris Agreement goals.',
                    'targets': ['13.1', '13.2', '13.3']
                },
                'SDG_INNOVATION': {
                    'sdg_number': 9,
                    'sdg_name': 'Industry, Innovation and Infrastructure',
                    'impact': 'Enhances investment in sustainable infrastructure and innovation. Promotes inclusive industrialization.',
                    'targets': ['9.1', '9.4', '9.5']
                }
            }
            return sdg_mapping.get(self.category, {})
        else:
            return {'sdg_aligned': False}