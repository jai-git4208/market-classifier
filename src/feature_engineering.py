import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator, ADXIndicator
# Note: VolumeSMAIndicator not available in ta library, we calculate volume features manually

class FeatureEngineer:
    """Create technical indicators and features for stock prediction"""
    
    def __init__(self):
        self.feature_columns = []
    
    def create_returns(self, df, col='Close', periods=[1, 3, 5, 7, 10]):
        """Calculate returns over multiple periods"""
        for period in periods:
            df[f'return_{period}d'] = df[col].pct_change(period)
            # Cap returns at ±100% to avoid extreme values
            df[f'return_{period}d'] = df[f'return_{period}d'].clip(-1.0, 1.0)
        return df
    
    def create_moving_averages(self, df, col='Close', windows=[5, 10, 20, 50]):
        """Calculate simple moving averages"""
        for window in windows:
            if len(df) >= window:
                df[f'sma_{window}'] = df[col].rolling(window=window).mean()
                # Avoid division by zero
                df[f'price_to_sma_{window}'] = df[col] / df[f'sma_{window}'].replace(0, np.nan)
                df[f'sma_{window}_slope'] = df[f'sma_{window}'].diff(5)
        return df
    
    def create_volatility_features(self, df, col='Close', windows=[10, 20, 30]):
        """Calculate volatility metrics"""
        returns = df[col].pct_change().clip(-1.0, 1.0)
        for window in windows:
            if len(df) >= window:
                df[f'volatility_{window}d'] = returns.rolling(window=window).std()
                df[f'rolling_max_{window}d'] = df[col].rolling(window=window).max()
                df[f'rolling_min_{window}d'] = df[col].rolling(window=window).min()
                # Avoid division by zero
                denominator = df[f'rolling_min_{window}d'].replace(0, np.nan)
                df[f'price_range_{window}d'] = (df[f'rolling_max_{window}d'] - df[f'rolling_min_{window}d']) / denominator
        return df
    
    def create_momentum_indicators(self, df, close_col='Close', high_col='High', low_col='Low'):
        """Calculate RSI, MACD, and other momentum indicators"""
        if len(df) < 26:
            return df
            
        try:
            # RSI
            rsi = RSIIndicator(close=df[close_col], window=14)
            df['rsi_14'] = rsi.rsi()
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            
            # MACD
            macd = MACD(close=df[close_col])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            df['macd_cross'] = ((df['macd'] > df['macd_signal']).astype(int) - 
                               (df['macd'] < df['macd_signal']).astype(int))
            
            # Rate of Change (capped)
            df['roc_10'] = ((df[close_col] - df[close_col].shift(10)) / df[close_col].shift(10).replace(0, np.nan)) * 100
            df['roc_10'] = df['roc_10'].clip(-50, 50)
            
            df['roc_20'] = ((df[close_col] - df[close_col].shift(20)) / df[close_col].shift(20).replace(0, np.nan)) * 100
            df['roc_20'] = df['roc_20'].clip(-50, 50)
            
            # Momentum
            df['momentum_10'] = df[close_col] - df[close_col].shift(10)
            
        except Exception as e:
            print(f"  Warning: Some momentum indicators failed: {e}")
        
        return df
    
    def create_bollinger_bands(self, df, col='Close', window=20):
        """Calculate Bollinger Bands"""
        if len(df) < window:
            return df
            
        try:
            bb = BollingerBands(close=df[col], window=window, window_dev=2)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            # Avoid division by zero
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid'].replace(0, np.nan)
            denominator = (df['bb_high'] - df['bb_low']).replace(0, np.nan)
            df['bb_position'] = (df[col] - df['bb_low']) / denominator
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
        except Exception as e:
            print(f"  Warning: Bollinger Bands failed: {e}")
        
        return df
    
    def create_volume_features(self, df, volume_col='Volume'):
        """Volume-based features"""
        if volume_col in df.columns and len(df) >= 20:
            df['volume_sma_20'] = df[volume_col].rolling(window=20).mean()
            # Avoid division by zero
            df['volume_ratio'] = df[volume_col] / df['volume_sma_20'].replace(0, np.nan)
            df['volume_change'] = df[volume_col].pct_change().clip(-1.0, 1.0)
            df['volume_trend'] = df['volume_sma_20'].diff(5)
            df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        return df
    
    def create_lag_features(self, df, col='Close', lags=[1, 2, 3, 5, 10]):
        """Create lagged price features"""
        for lag in lags:
            if len(df) > lag:
                df[f'close_lag_{lag}'] = df[col].shift(lag)
                df[f'return_lag_{lag}'] = df[col].pct_change().shift(lag).clip(-1.0, 1.0)
        return df
    
    def create_advanced_indicators(self, df, close_col='Close', high_col='High', low_col='Low', volume_col='Volume'):
        """Create advanced technical indicators"""
        if len(df) < 30:
            return df
        
        try:
            # ADX (Average Directional Index) - trend strength
            if high_col in df.columns and low_col in df.columns:
                adx = ADXIndicator(high=df[high_col], low=df[low_col], close=df[close_col], window=14)
                df['adx'] = adx.adx()
                df['adx_strong_trend'] = (df['adx'] > 25).astype(int)
            
            # Stochastic Oscillator
            if high_col in df.columns and low_col in df.columns:
                stoch = StochasticOscillator(high=df[high_col], low=df[low_col], close=df[close_col])
                df['stoch_k'] = stoch.stoch()
                df['stoch_d'] = stoch.stoch_signal()
                df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
                df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
            
            # Williams %R
            if high_col in df.columns and low_col in df.columns:
                williams = WilliamsRIndicator(high=df[high_col], low=df[low_col], close=df[close_col])
                df['williams_r'] = williams.williams_r()
            
            # Average True Range (ATR) - volatility measure
            if high_col in df.columns and low_col in df.columns:
                atr = AverageTrueRange(high=df[high_col], low=df[low_col], close=df[close_col], window=14)
                df['atr'] = atr.average_true_range()
                df['atr_percent'] = (df['atr'] / df[close_col].replace(0, np.nan)) * 100
                df['atr_percent'] = df['atr_percent'].clip(0, 50)
            
            # EMA indicators
            if len(df) >= 50:
                ema_12 = EMAIndicator(close=df[close_col], window=12)
                ema_26 = EMAIndicator(close=df[close_col], window=26)
                df['ema_12'] = ema_12.ema_indicator()
                df['ema_26'] = ema_26.ema_indicator()
                df['ema_cross'] = ((df['ema_12'] > df['ema_26']).astype(int) - 
                                  (df['ema_12'] < df['ema_26']).astype(int))
            
        except Exception as e:
            print(f"  Warning: Some advanced indicators failed: {e}")
        
        return df
    
    def create_time_features(self, df):
        """Create time-based features"""
        if df.index.dtype == 'datetime64[ns]' or isinstance(df.index[0], pd.Timestamp):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        return df
    
    def create_correlation_features(self, df, ticker_dfs_dict, close_col='Close'):
        """Create correlation features between tickers"""
        if not ticker_dfs_dict or len(ticker_dfs_dict) < 2:
            return df
        
        try:
            # Calculate rolling correlation with other tickers
            current_returns = df[close_col].pct_change().clip(-1.0, 1.0)
            current_ticker = close_col.split('_')[0] if '_' in close_col else None
            
            for other_ticker, other_df in ticker_dfs_dict.items():
                if other_ticker != current_ticker and 'Close' in other_df.columns:
                    other_returns = other_df['Close'].pct_change().clip(-1.0, 1.0)
                    
                    # Align indices
                    common_idx = current_returns.index.intersection(other_returns.index)
                    if len(common_idx) >= 20:
                        aligned_current = current_returns.loc[common_idx]
                        aligned_other = other_returns.loc[common_idx]
                        
                        # Rolling correlation
                        corr_20 = aligned_current.rolling(20).corr(aligned_other)
                        df[f'corr_20d_{other_ticker}'] = corr_20.reindex(df.index)
                        
                        if len(common_idx) >= 60:
                            corr_60 = aligned_current.rolling(60).corr(aligned_other)
                            df[f'corr_60d_{other_ticker}'] = corr_60.reindex(df.index)
        except Exception as e:
            print(f"  Warning: Correlation features failed: {e}")
        
        return df
    
    def create_market_regime_features(self, df, market_data=None):
        """Create market regime features (SPY correlation, market volatility)"""
        if market_data is None or market_data.empty:
            return df
        
        try:
            # Calculate correlation with market (SPY)
            if 'SPY_Close' in market_data.columns:
                spy_returns = market_data['SPY_Close'].pct_change().clip(-1.0, 1.0)
                stock_returns = df['Close'].pct_change().clip(-1.0, 1.0)
                
                # Align indices
                common_idx = stock_returns.index.intersection(spy_returns.index)
                if len(common_idx) >= 20:
                    aligned_stock = stock_returns.loc[common_idx]
                    aligned_spy = spy_returns.loc[common_idx]
                    
                    # Rolling correlation with market
                    corr_20 = aligned_stock.rolling(20).corr(aligned_spy)
                    df['market_corr_20d'] = corr_20.reindex(df.index)
                    
                    # Beta approximation (simplified)
                    if len(common_idx) >= 60:
                        stock_cov = aligned_stock.rolling(60).cov(aligned_spy)
                        spy_var = aligned_spy.rolling(60).var()
                        beta_60 = stock_cov / spy_var.replace(0, np.nan)
                        df['beta_60d'] = beta_60.reindex(df.index)
                
                # Market volatility
                if len(spy_returns) >= 20:
                    market_vol = spy_returns.rolling(20).std()
                    df['market_volatility_20d'] = market_vol.reindex(df.index)
        except Exception as e:
            print(f"  Warning: Market regime features failed: {e}")
        
        return df
    
    def create_interaction_features(self, df, close_col='Close'):
        """Create feature interactions"""
        if 'rsi_14' in df.columns and 'volatility_20d' in df.columns:
            df['rsi_vol_interaction'] = df['rsi_14'] * df['volatility_20d']
        
        if 'return_1d' in df.columns and 'volume_ratio' in df.columns:
            df['return_volume_interaction'] = df['return_1d'] * df['volume_ratio']
        
        # Price momentum interactions
        if 'rsi_14' in df.columns and 'macd' in df.columns:
            df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
        
        # Volatility-price interactions
        if 'volatility_20d' in df.columns and 'return_1d' in df.columns:
            df['vol_return_interaction'] = df['volatility_20d'] * abs(df['return_1d'])
        
        # Additional advanced interactions
        if 'rsi_14' in df.columns and 'bb_position' in df.columns:
            df['rsi_bb_interaction'] = df['rsi_14'] * df['bb_position']
        
        if 'macd' in df.columns and 'volume_ratio' in df.columns:
            df['macd_volume_interaction'] = df['macd'] * df['volume_ratio']
        
        if 'adx' in df.columns and 'volatility_20d' in df.columns:
            df['adx_vol_interaction'] = df['adx'] * df['volatility_20d']
        
        return df
    
    def create_fibonacci_features(self, df, col='Close', window=20):
        """Create Fibonacci retracement levels (advanced feature)"""
        if len(df) < window:
            return df
        
        try:
            rolling_high = df[col].rolling(window=window).max()
            rolling_low = df[col].rolling(window=window).min()
            price_range = rolling_high - rolling_low
            
            # Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
            df['fib_236'] = rolling_low + price_range * 0.236
            df['fib_382'] = rolling_low + price_range * 0.382
            df['fib_500'] = rolling_low + price_range * 0.500
            df['fib_618'] = rolling_low + price_range * 0.618
            df['fib_786'] = rolling_low + price_range * 0.786
            
            # Distance to Fibonacci levels
            df['dist_to_fib_236'] = (df[col] - df['fib_236']) / df[col].replace(0, np.nan)
            df['dist_to_fib_382'] = (df[col] - df['fib_382']) / df[col].replace(0, np.nan)
            df['dist_to_fib_500'] = (df[col] - df['fib_500']) / df[col].replace(0, np.nan)
            df['dist_to_fib_618'] = (df[col] - df['fib_618']) / df[col].replace(0, np.nan)
            df['dist_to_fib_786'] = (df[col] - df['fib_786']) / df[col].replace(0, np.nan)
        except Exception as e:
            print(f"  Warning: Fibonacci features failed: {e}")
        
        return df
    
    def create_ichimoku_features(self, df, close_col='Close', high_col='High', low_col='Low'):
        """Create Ichimoku Cloud features (advanced Japanese technical analysis)"""
        if len(df) < 52 or high_col not in df.columns or low_col not in df.columns:
            return df
        
        try:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            period1_high = df[high_col].rolling(window=9).max()
            period1_low = df[low_col].rolling(window=9).min()
            df['ichimoku_tenkan'] = (period1_high + period1_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            period2_high = df[high_col].rolling(window=26).max()
            period2_low = df[low_col].rolling(window=26).min()
            df['ichimoku_kijun'] = (period2_high + period2_low) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods
            df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods
            period3_high = df[high_col].rolling(window=52).max()
            period3_low = df[low_col].rolling(window=52).min()
            df['ichimoku_senkou_b'] = ((period3_high + period3_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Close price shifted -26 periods
            df['ichimoku_chikou'] = df[close_col].shift(-26)
            
            # Cloud position features
            df['ichimoku_above_cloud'] = ((df[close_col] > df['ichimoku_senkou_a']) & 
                                          (df[close_col] > df['ichimoku_senkou_b'])).astype(int)
            df['ichimoku_below_cloud'] = ((df[close_col] < df['ichimoku_senkou_a']) & 
                                          (df[close_col] < df['ichimoku_senkou_b'])).astype(int)
        except Exception as e:
            print(f"  Warning: Ichimoku features failed: {e}")
        
        return df
    
    def create_all_features(self, df, ticker_prefix='', market_data=None, ticker_dfs_dict=None):
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
        df = self.create_advanced_indicators(df, close_col, high_col, low_col, volume_col)
        df = self.create_lag_features(df, col=close_col)
        df = self.create_time_features(df)
        df = self.create_interaction_features(df, close_col)
        
        # Advanced features (Fibonacci, Ichimoku)
        if len(df) >= 52:
            df = self.create_fibonacci_features(df, col=close_col)
            df = self.create_ichimoku_features(df, close_col, high_col, low_col)
        
        # Market regime features (if market data available)
        if market_data is not None:
            df = self.create_market_regime_features(df, market_data)
        
        # Correlation features (if multiple tickers available)
        if ticker_dfs_dict is not None:
            df = self.create_correlation_features(df, ticker_dfs_dict, close_col)
        
        # Final cleanup - replace inf and extreme values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def create_target_label(self, df, target_ticker, forward_days=1):
        """Create binary target WITHOUT DATA LEAKAGE"""
        # Handle both prefixed (AAPL_Close) and non-prefixed (Close) column names
        if target_ticker:
            close_col = f'{target_ticker}_Close'
        else:
            close_col = 'Close'
        
        # Fallback if exact match not found
        if close_col not in df.columns:
            # Try to find Close column
            if 'Close' in df.columns:
                close_col = 'Close'
            else:
                # Find any column ending with Close
                close_cols = [col for col in df.columns if col.endswith('_Close') or col == 'Close']
                if close_cols:
                    close_col = close_cols[0]
                else:
                    raise ValueError(f"Could not find Close column in dataframe. Available columns: {list(df.columns)}")
        
        df['future_close'] = df[close_col].shift(-forward_days)
        df['target'] = (df['future_close'] > df[close_col]).astype(int)
        
        df = df[:-forward_days]
        
        return df
    
    def create_target_label_significant(self, df, target_ticker, threshold=0.02):
        """Create target for significant movements only"""
        # Handle both prefixed (AAPL_Close) and non-prefixed (Close) column names
        if target_ticker:
            close_col = f'{target_ticker}_Close'
        else:
            close_col = 'Close'
        
        # Fallback if exact match not found
        if close_col not in df.columns:
            # Try to find Close column
            if 'Close' in df.columns:
                close_col = 'Close'
            else:
                # Find any column ending with Close
                close_cols = [col for col in df.columns if col.endswith('_Close') or col == 'Close']
                if close_cols:
                    close_col = close_cols[0]
                else:
                    raise ValueError(f"Could not find Close column in dataframe. Available columns: {list(df.columns)}")
        
        df['future_close'] = df[close_col].shift(-1)
        df['return'] = (df['future_close'] - df[close_col]) / df[close_col].replace(0, np.nan)
        
        # Only keep significant movements
        df['target'] = np.where(df['return'] > threshold, 1,
                       np.where(df['return'] < -threshold, 0, np.nan))
        
        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)
        
        return df