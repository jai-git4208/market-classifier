import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator
import json
import os

class FeatureEngineer:
    """Create technical indicators and features for stock prediction"""
    
    def __init__(self):
        self.feature_columns = []
    
    def create_returns(self, df, col='Close', periods=[1, 3, 5, 7]):
        """Calculate returns over multiple periods"""
        for period in periods:
            df[f'return_{period}d'] = df[col].pct_change(period)
        return df
    
    def create_moving_averages(self, df, col='Close', windows=[5, 10, 20]):
        """Calculate simple moving averages"""
        for window in windows:
            df[f'sma_{window}'] = df[col].rolling(window=window).mean()
            df[f'price_to_sma_{window}'] = df[col] / df[f'sma_{window}']
        return df
    
    def create_volatility_features(self, df, col='Close', window=14):
        """Calculate volatility metrics"""
        returns = df[col].pct_change()
        df[f'volatility_{window}d'] = returns.rolling(window=window).std()
        df[f'rolling_max_{window}d'] = df[col].rolling(window=window).max()
        df[f'rolling_min_{window}d'] = df[col].rolling(window=window).min()
        df[f'price_range_{window}d'] = (df[f'rolling_max_{window}d'] - df[f'rolling_min_{window}d']) / df[f'rolling_min_{window}d']
        return df
    
    def create_momentum_indicators(self, df, close_col='Close', high_col='High', low_col='Low'):
        """Calculate RSI, MACD, and other momentum indicators"""
        # RSI
        rsi = RSIIndicator(close=df[close_col], window=14)
        df['rsi_14'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=df[close_col])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Rate of Change
        df['roc_10'] = ((df[close_col] - df[close_col].shift(10)) / df[close_col].shift(10)) * 100
        
        return df
    
    def create_bollinger_bands(self, df, col='Close', window=20):
        """Calculate Bollinger Bands"""
        bb = BollingerBands(close=df[col], window=window, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df[col] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        return df
    
    def create_volume_features(self, df, volume_col='Volume'):
        """Volume-based features"""
        if volume_col in df.columns:
            df['volume_sma_20'] = df[volume_col].rolling(window=20).mean()
            df['volume_ratio'] = df[volume_col] / df['volume_sma_20']
            df['volume_change'] = df[volume_col].pct_change()
        return df
    
    def create_lag_features(self, df, col='Close', lags=[1, 2, 3, 5]):
        """Create lagged price features"""
        for lag in lags:
            df[f'close_lag_{lag}'] = df[col].shift(lag)
        return df
    
    def create_all_features(self, df, ticker_prefix=''):
        """Create all features for a single ticker"""
        close_col = f'{ticker_prefix}Close' if ticker_prefix else 'Close'
        high_col = f'{ticker_prefix}High' if ticker_prefix else 'High'
        low_col = f'{ticker_prefix}Low' if ticker_prefix else 'Low'
        volume_col = f'{ticker_prefix}Volume' if ticker_prefix else 'Volume'
        
        # Check if required columns exist
        if close_col not in df.columns:
            print(f"Warning: {close_col} not found, skipping {ticker_prefix}")
            return df
        
        df = self.create_returns(df, col=close_col)
        df = self.create_moving_averages(df, col=close_col)
        df = self.create_volatility_features(df, col=close_col)
        
        # Only create momentum indicators if High/Low exist
        if high_col in df.columns and low_col in df.columns:
            df = self.create_momentum_indicators(df, close_col, high_col, low_col)
        else:
            print(f"Warning: High/Low not found for {ticker_prefix}, skipping momentum indicators")
        
        df = self.create_bollinger_bands(df, col=close_col)
        df = self.create_volume_features(df, volume_col)
        df = self.create_lag_features(df, col=close_col)
        
        return df
    
    def create_target_label(self, df, target_ticker, forward_days=1):
        """
        Create binary target WITHOUT DATA LEAKAGE
        Target = 1 if next day's close > today's close, else 0
        """
        close_col = f'{target_ticker}_Close'
        
        if close_col not in df.columns:
            raise ValueError(f"Target ticker column {close_col} not found in dataframe")
        
        # Shift future price BACKWARDS to align with current features
        df['future_close'] = df[close_col].shift(-forward_days)
        df['target'] = (df['future_close'] > df[close_col]).astype(int)
        
        # Remove last row(s) that don't have future data
        df = df[:-forward_days]
        
        return df


def main():
    """Main feature engineering pipeline with user configuration"""
    print("\n" + "="*60)
    print("   FEATURE ENGINEERING - UNIVERSAL STOCK PREDICTOR")
    print("="*60)
    
    # Load configuration
    config_path = 'data/config.json'
    if not os.path.exists(config_path):
        print("\n❌ Error: config.json not found!")
        print("Please run the data loader script first.")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tickers = config['tickers']
    target_ticker = config['target_ticker']
    
    print(f"\n📊 Configuration loaded:")
    print(f"   • Tickers: {', '.join(tickers)}")
    print(f"   • Target: {target_ticker}")
    print(f"   • Data downloaded: {config['download_date']}")
    
    # Load combined data
    data_path = 'data/combined_stock_data.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ Error: {data_path} not found!")
        print("Please run the data loader script first.")
        return None
    
    print(f"\n⏳ Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Create features for each ticker
    print("\n🔧 Creating technical indicators and features...")
    for ticker in tickers:
        print(f"   • Processing {ticker}...", end=" ")
        try:
            df = fe.create_all_features(df, ticker_prefix=f'{ticker}_')
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Create target label
    print(f"\n🎯 Creating prediction target for {target_ticker}...")
    
    # Ask user for prediction horizon
    print("\nPrediction Horizon:")
    print("  1. Next day (1 day ahead) - Recommended")
    print("  2. 2 days ahead")
    print("  3. 3 days ahead")
    print("  4. 1 week ahead (5 days)")
    
    horizon_map = {'1': 1, '2': 2, '3': 3, '4': 5}
    horizon_choice = input("\nSelect horizon (1-4, default=1): ").strip() or '1'
    forward_days = horizon_map.get(horizon_choice, 1)
    
    print(f"\n⏩ Predicting {forward_days} day(s) ahead")
    
    try:
        df = fe.create_target_label(df, target_ticker, forward_days=forward_days)
    except Exception as e:
        print(f"\n❌ Error creating target: {e}")
        return None
    
    # Remove rows with NaN values
    print("\n🧹 Cleaning data (removing NaN values)...")
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    print(f"   • Rows before: {initial_rows}")
    print(f"   • Rows after: {final_rows}")
    print(f"   • Removed: {initial_rows - final_rows}")
    
    # Display target distribution
    target_counts = df['target'].value_counts()
    print(f"\n📊 Target Distribution:")
    print(f"   • DOWN (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   • UP (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Save processed data
    output_path = 'data/processed_features.csv'
    df.to_csv(output_path)
    print(f"\n💾 Processed data saved to: {output_path}")
    
    # Update config
    config['forward_days'] = forward_days
    config['feature_engineering_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    config['final_rows'] = final_rows
    config['total_features'] = len(df.columns)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print(f"   • Total features: {len(df.columns)}")
    print(f"   • Training samples: {final_rows}")
    print("\n📈 READY FOR MODEL TRAINING")
    print("   Run model_training.py to train the prediction model")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run feature engineering
    processed_data = main()