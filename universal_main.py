import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class UniversalStockDataLoader:
    """Download and prepare stock data for ANY ticker"""
    
    def __init__(self, tickers=None):
        """
        Initialize with list of tickers
        tickers: list of stock symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        """
        if tickers is None:
            tickers = []
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.raw_data = None
        self.processed_data = None
        
    def download_data(self, period='3y', interval='1d', max_retries=3):
        """Download latest data from Yahoo Finance with retry logic"""
        if not self.tickers:
            raise ValueError("No tickers specified. Please add tickers first.")
            
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
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        # If all retries fail, try downloading individually
        print("\nTrying individual ticker downloads...")
        return self._download_individually(period, interval)
    
    def _download_individually(self, period='3y', interval='1d'):
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
            raise ValueError("Failed to download data for any ticker. Please check ticker symbols and internet connection.")
        
        # Combine into multi-index DataFrame
        if len(all_data) == 1:
            # Single ticker - flatten structure
            ticker = list(all_data.keys())[0]
            combined = all_data[ticker]
            combined.columns = [f'{ticker}_{col}' for col in combined.columns]
        else:
            # Multiple tickers
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
                df.columns = [col.replace(f'{ticker}_', '') for col in df.columns]
            
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


def get_user_tickers():
    """Interactive function to get tickers from user"""
    print("\n" + "="*60)
    print("   UNIVERSAL STOCK PREDICTION SYSTEM")
    print("="*60)
    print("\nThis system can predict price movements for ANY stock ticker!")
    print("\nExamples of valid tickers:")
    print("  • Tech: AAPL, MSFT, GOOGL, TSLA, NVDA")
    print("  • Finance: JPM, BAC, GS, V, MA")
    print("  • Energy: XOM, CVX, COP")
    print("  • ETFs: SPY, QQQ, DIA, ICLN, TAN")
    print("  • Crypto: BTC-USD, ETH-USD")
    
    while True:
        print("\n" + "-"*60)
        ticker_input = input("\nEnter ticker symbol(s) to predict (comma-separated): ").strip().upper()
        
        if not ticker_input:
            print("❌ Please enter at least one ticker symbol")
            continue
        
        # Parse tickers
        tickers = [t.strip() for t in ticker_input.split(',') if t.strip()]
        
        if not tickers:
            print("❌ No valid tickers found")
            continue
        
        print(f"\n📊 You entered: {', '.join(tickers)}")
        
        # Ask which ticker to predict
        if len(tickers) == 1:
            target_ticker = tickers[0]
            print(f"\n🎯 Target ticker for prediction: {target_ticker}")
        else:
            print("\n🎯 Which ticker do you want to predict?")
            for i, ticker in enumerate(tickers, 1):
                print(f"   {i}. {ticker}")
            
            while True:
                choice = input(f"\nEnter number (1-{len(tickers)}): ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(tickers):
                        target_ticker = tickers[idx]
                        break
                    else:
                        print(f"❌ Please enter a number between 1 and {len(tickers)}")
                except ValueError:
                    print("❌ Please enter a valid number")
        
        # Confirm
        print(f"\n{'='*60}")
        print(f"📈 Tickers to download: {', '.join(tickers)}")
        print(f"🎯 Target for prediction: {target_ticker}")
        print(f"{'='*60}")
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm == 'y':
            return tickers, target_ticker
        else:
            print("\n🔄 Let's try again...")


def main():
    """Main execution function with user input"""
    # Get tickers from user
    tickers, target_ticker = get_user_tickers()
    
    # Initialize data loader
    loader = UniversalStockDataLoader(tickers=tickers)
    
    # Get data period from user
    print("\n" + "-"*60)
    print("Data Period Options:")
    print("  1. 1 month (1mo)")
    print("  2. 3 months (3mo)")
    print("  3. 6 months (6mo)")
    print("  4. 1 year (1y)")
    print("  5. 2 years (2y)")
    print("  6. 3 years (3y) - Recommended")
    print("  7. 5 years (5y)")
    
    period_map = {
        '1': '1mo', '2': '3mo', '3': '6mo', 
        '4': '1y', '5': '2y', '6': '3y', '7': '5y'
    }
    
    period_choice = input("\nSelect period (1-7, default=6): ").strip() or '6'
    period = period_map.get(period_choice, '3y')
    
    print(f"\n⏳ Downloading {period} of historical data...")
    
    try:
        # Download data
        loader.download_data(period=period)
        
        # Combine all tickers
        print("\n🔧 Processing and combining ticker data...")
        combined_df = loader.combine_all_tickers()
        
        print(f"\n✅ Data preparation complete!")
        print(f"   • Total days: {len(combined_df)}")
        print(f"   • Date range: {combined_df.index[0].date()} to {combined_df.index[-1].date()}")
        print(f"   • Features: {len(combined_df.columns)}")
        
        # Save to CSV for next steps
        output_file = 'data/combined_stock_data.csv'
        combined_df.to_csv(output_file)
        print(f"\n💾 Data saved to: {output_file}")
        
        # Save configuration
        config = {
            'tickers': tickers,
            'target_ticker': target_ticker,
            'period': period,
            'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open('data/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Configuration saved to: data/config.json")
        
        print("\n" + "="*60)
        print("✅ READY FOR NEXT STEPS:")
        print("   1. Run universal_feature.py to create indicators")
        print("   2. Run universal_train.py to train prediction model")
        print("="*60)
        
        return combined_df, target_ticker
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("  • Ticker symbols are correct")
        print("  • Internet connection is stable")
        print("  • Yahoo Finance is accessible")
        return None, None


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run main function
    data, target = main()