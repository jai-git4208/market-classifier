"""
Backtesting Framework for Market Movement Classifier
Tests model performance on historical data with realistic trading simulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class Backtester:
    """Backtesting framework for evaluating trading strategies"""
    
    def __init__(self, model, scaler, feature_names, initial_capital=10000):
        """
        Initialize backtester
        
        Args:
            model: Trained XGBoost model
            scaler: Fitted StandardScaler
            feature_names: List of feature names
            initial_capital: Starting capital in dollars
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.initial_capital = initial_capital
        self.results = None
    
    def run_backtest(self, df, price_col, target_col='target', confidence_threshold=0.6):
        """
        Run backtest on historical data
        
        Args:
            df: DataFrame with features, prices, and targets
            price_col: Column name for current price
            target_col: Column name for target labels
            confidence_threshold: Minimum confidence to take a trade
        
        Returns:
            Dictionary with backtest results
        """
        # Prepare features
        exclude_cols = [target_col, 'future_close', 'return'] + \
                      [col for col in df.columns if 'Date' in col or 'date' in col.lower()]
        feature_cols = [col for col in df.columns if col not in exclude_cols and col in self.feature_names]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        prices = df[price_col].copy()
        
        # Clean data
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill().fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Initialize tracking
        capital = self.initial_capital
        positions = []  # List of (date, action, price, capital)
        equity_curve = [capital]
        trades = []
        
        position = None  # Current position: None, 'LONG', or 'SHORT'
        entry_price = None
        
        for i in range(len(df)):
            current_price = prices.iloc[i]
            pred = predictions[i]
            prob = probabilities[i]
            actual = y.iloc[i] if i < len(y) else None
            
            # Only trade if confidence is high enough
            if prob >= confidence_threshold or prob <= (1 - confidence_threshold):
                confidence = max(prob, 1 - prob)
                
                # Close existing position if direction changes
                if position is not None:
                    if (position == 'LONG' and pred == 0) or (position == 'SHORT' and pred == 1):
                        # Close position
                        if position == 'LONG':
                            pnl = (current_price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - current_price) / entry_price
                        
                        capital = capital * (1 + pnl)
                        
                        trades.append({
                            'entry_date': df.index[positions[-1][0]] if positions else df.index[i],
                            'exit_date': df.index[i],
                            'action': f'CLOSE_{position}',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'capital': capital,
                            'correct': (position == 'LONG' and actual == 1) or (position == 'SHORT' and actual == 0)
                        })
                        
                        position = None
                        entry_price = None
                
                # Open new position
                if position is None:
                    if pred == 1:  # Predict UP
                        position = 'LONG'
                        entry_price = current_price
                    else:  # Predict DOWN
                        position = 'SHORT'
                        entry_price = current_price
                    
                    positions.append((i, position, current_price, capital))
            
            equity_curve.append(capital)
        
        # Close final position if exists
        if position is not None and len(df) > 0:
            final_price = prices.iloc[-1]
            if position == 'LONG':
                pnl = (final_price - entry_price) / entry_price
            else:
                pnl = (entry_price - final_price) / entry_price
            
            capital = capital * (1 + pnl)
            equity_curve[-1] = capital
            
            trades.append({
                'entry_date': df.index[positions[-1][0]] if positions else df.index[0],
                'exit_date': df.index[-1],
                'action': f'CLOSE_{position}',
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl': pnl,
                'capital': capital,
                'correct': None
            })
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        # Fix: equity_curve has len(df)+1 elements (initial + one per row), so align with df.index
        if len(equity_curve) == len(df) + 1:
            equity_series = pd.Series(equity_curve[1:], index=df.index)  # Skip initial capital
        else:
            equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
        
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(equity_series)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Win rate
        if trades:
            correct_trades = [t for t in trades if t.get('correct') is True]
            win_rate = len(correct_trades) / len([t for t in trades if t.get('correct') is not None]) if trades else 0
        else:
            win_rate = 0
        
        self.results = {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'equity_curve': equity_series,
            'trades': trades
        }
        
        return self.results
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0
        
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
    
    def plot_results(self, save_path='results/backtest_results.png'):
        """Plot backtest results"""
        if self.results is None:
            print("  ⚠️  No backtest results to plot. Run backtest first.")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Equity curve
        equity = self.results['equity_curve']
        axes[0, 0].plot(equity.index, equity.values, linewidth=2, color='#3b82f6')
        axes[0, 0].axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        axes[0, 0].set_title(f'Equity Curve\nTotal Return: {self.results["total_return_pct"]:.2f}%')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[0, 1].set_title(f'Drawdown\nMax Drawdown: {self.results["max_drawdown_pct"]:.2f}%')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(alpha=0.3)
        
        # Trade P&L distribution
        if self.results['trades']:
            pnls = [t['pnl'] * 100 for t in self.results['trades']]
            axes[1, 0].hist(pnls, bins=20, color='#10b981', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_title(f'Trade P&L Distribution\nWin Rate: {self.results["win_rate"]*100:.1f}%')
            axes[1, 0].set_xlabel('P&L (%)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(alpha=0.3)
        
        # Performance metrics
        metrics_text = f"""
        Initial Capital: ${self.results['initial_capital']:,.2f}
        Final Capital: ${self.results['final_capital']:,.2f}
        Total Return: {self.results['total_return_pct']:.2f}%
        Sharpe Ratio: {self.results['sharpe_ratio']:.2f}
        Max Drawdown: {self.results['max_drawdown_pct']:.2f}%
        Number of Trades: {self.results['num_trades']}
        Win Rate: {self.results['win_rate']*100:.1f}%
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Metrics')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Backtest results saved to {save_path}")
    
    def print_summary(self):
        """Print backtest summary"""
        if self.results is None:
            print("  ⚠️  No backtest results. Run backtest first.")
            return
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        print(f"Initial Capital:     ${self.results['initial_capital']:,.2f}")
        print(f"Final Capital:       ${self.results['final_capital']:,.2f}")
        print(f"Total Return:        {self.results['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio:        {self.results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:        {self.results['max_drawdown_pct']:.2f}%")
        print(f"Number of Trades:    {self.results['num_trades']}")
        print(f"Win Rate:            {self.results['win_rate']*100:.1f}%")
        print("="*70)

