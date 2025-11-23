import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator

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
        
        df = self.create_returns(df, col=close_col)
        df = self.create_moving_averages(df, col=close_col)
        df = self.create_volatility_features(df, col=close_col)
        df = self.create_momentum_indicators(df, close_col, high_col, low_col)
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
        
        # Shift future price BACKWARDS to align with current features
        df['future_close'] = df[close_col].shift(-forward_days)
        df['target'] = (df['future_close'] > df[close_col]).astype(int)
        
        # Remove last row(s) that don't have future data
        df = df[:-forward_days]
        
        return df