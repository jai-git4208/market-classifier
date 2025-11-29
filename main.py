import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

import xgboost as xgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
warnings.filterwarnings('ignore')

from data_loader import CleanEnergyDataLoader
from feature_engineering import FeatureEngineer
from model_training import XGBoostMarketClassifier

MACRO_SYMBOLS = {'^VIX': 'vix', '^TNX': 'tenyr'}
MACRO_LOOKBACK_DAYS = 120

def ensure_datetime_index(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df.index = pd.to_datetime(df['Date'], errors='coerce')
            df = df.drop(columns=['Date'], errors='ignore')
        elif 'date' in df.columns:
            df.index = pd.to_datetime(df['date'], errors='coerce')
            df = df.drop(columns=['date'], errors='ignore')
        else:
            df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    return df

def fetch_macro_indicators(index, lookback=MACRO_LOOKBACK_DAYS):
    try:
        import yfinance as yf
        start = (index.min() - pd.Timedelta(days=lookback)).date()
        end = (index.max() + pd.Timedelta(days=2)).date()
        frames = []
        for symbol, label in MACRO_SYMBOLS.items():
            data = yf.download(symbol, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=True)
            if data.empty:
                continue
            frames.append(data['Close'].rename(label))
        if not frames:
            return None
        macro_df = pd.concat(frames, axis=1)
        macro_df = macro_df.reindex(index).ffill().bfill().fillna(method='ffill').fillna(0)
        return macro_df
    except Exception:
        return None

def augment_with_macro_indicators(df):
    df = ensure_datetime_index(df)
    macro_df = fetch_macro_indicators(df.index)
    if macro_df is None or macro_df.empty:
        return df
    for col in macro_df.columns:
        df[f'macro_{col}_close'] = macro_df[col].values
        df[f'macro_{col}_ma5'] = macro_df[col].rolling(window=5, min_periods=1).mean().values
    if 'vix' in macro_df.columns:
        df['macro_vix_pctchg'] = macro_df['vix'].pct_change().fillna(0).values
    else:
        df['macro_vix_pctchg'] = np.zeros(len(df))
    if 'vix' in macro_df.columns and 'tenyr' in macro_df.columns:
        ratio = macro_df['vix'] / macro_df['tenyr'].replace(0, np.nan)
        df['macro_vix_to_tenyr'] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0).values
    return df

def train_single_ticker(ticker, market_data=None, use_significant_moves=False):
    import yfinance as yf
    from data_loader import CleanEnergyDataLoader
    from feature_engineering import FeatureEngineer
    from model_training import XGBoostMarketClassifier
    
    try:
        loader = CleanEnergyDataLoader(category='CUSTOM', custom_tickers=[ticker])
        raw_data = loader.download_data(period='5y', interval='1d')
        
        if raw_data.empty or len(raw_data) < 50:
            return None
        
        engineer = FeatureEngineer()
        ticker_df = loader.get_ticker_data(ticker)
        
        if ticker_df.empty or len(ticker_df) < 50:
            return None
        
        ticker_df = engineer.create_all_features(
            ticker_df,
            ticker_prefix='',
            market_data=market_data,
            ticker_dfs_dict=None
        )
        
        if use_significant_moves:
            ticker_df = engineer.create_target_label_significant(ticker_df, target_ticker='', threshold=0.02)
        else:
            ticker_df = engineer.create_target_label(ticker_df, target_ticker='', forward_days=1)
        
        if len(ticker_df) < 50:
            return None
        
        use_feature_selection = ticker_df.shape[1] > 100
        top_k = min(150, ticker_df.shape[1] - 2)
        
        classifier = XGBoostMarketClassifier(
            random_state=42,
            use_feature_selection=use_feature_selection,
            top_k_features=top_k
        )
        X_train, X_test, y_train, y_test = classifier.prepare_data(ticker_df)
        
        if len(X_train) < 30:
            return None
        
        use_tuning = len(X_train) > 100
        classifier.train(X_train, y_train, use_tuning=use_tuning)
        metrics = classifier.evaluate(X_train, X_test, y_train, y_test)
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            current_price = float(hist['Close'].iloc[-1]) if not hist.empty else None
        except:
            current_price = None
        
        exclude_cols = ['target', 'future_close'] + (['return'] if use_significant_moves else [])
        latest_features = ticker_df.drop(columns=[col for col in exclude_cols if col in ticker_df.columns]).iloc[[-1]]
        next_day_pred = classifier.predict_next_day(latest_features, current_price=current_price)
        
        return {
            'ticker': ticker,
            'classifier': classifier,
            'metrics': metrics,
            'prediction': next_day_pred,
            'current_price': current_price,
            'ticker_df': ticker_df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'raw_data': raw_data
        }
    except Exception as e:
        print(f"  ✗ Error training {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(category='SDG_CLEAN_ENERGY', custom_tickers=None, use_significant_moves=False):
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("="*70)
    print("MULTI-CATEGORY MARKET MOVEMENT CLASSIFIER")
    print("="*70)
    
    print("\n[1/6] Loading Data...")
    
    if custom_tickers:
        loader = CleanEnergyDataLoader(category='CUSTOM', custom_tickers=custom_tickers)
    else:
        loader = CleanEnergyDataLoader(category=category)
    
    sdg_info = loader.get_sdg_info()
    if sdg_info.get('sdg_aligned'):
        print(f"\n🌍 SDG #{sdg_info['sdg_number']}: {sdg_info['sdg_name']}")
        print(f"Impact: {sdg_info['impact']}")
    
    try:
        raw_data = loader.download_data(period='5y', interval='1d')
        
        if raw_data.empty or len(raw_data) < 50:
            raise ValueError("Insufficient data downloaded")
            
    except Exception as e:
        print(f"\n⚠️  Download failed: {e}")
        print("Please check internet connection or try different tickers.")
        return None
    
    print("\n[2/6] Engineering Features...")
    engineer = FeatureEngineer()
    
    market_data = None
    try:
        import yfinance as yf
        spy_data = yf.download('SPY', period='5y', interval='1d', progress=False, auto_adjust=True)
        if not spy_data.empty:
            spy_data.columns = [f'SPY_{col}' for col in spy_data.columns]
            market_data = spy_data
            print("  ✓ Market data downloaded")
    except Exception as e:
        print(f"  ⚠️  Could not download market data: {e}")
    
    ticker_dfs_dict = {}
    all_ticker_features = []
    
    for ticker in loader.tickers:
        print(f"  Processing {ticker}...")
        try:
            ticker_df = loader.get_ticker_data(ticker)
            
            if ticker_df.empty or len(ticker_df) < 50:
                print(f"  ⚠️  Insufficient data for {ticker}, skipping...")
                continue
            
            ticker_dfs_dict[ticker] = ticker_df.copy()
                
            ticker_df = engineer.create_all_features(
                ticker_df, 
                ticker_prefix='',
                market_data=market_data,
                ticker_dfs_dict={k: v for k, v in ticker_dfs_dict.items() if k != ticker}
            )
            ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
            all_ticker_features.append(ticker_df)
            
        except Exception as e:
            print(f"  ✗ Error processing {ticker}: {e}")
            continue
    
    if not all_ticker_features:
        print("\n❌ No ticker data was successfully processed!")
        return None
    
    combined_df = pd.concat(all_ticker_features, axis=1)
    combined_df = combined_df.dropna()
    combined_df = augment_with_macro_indicators(combined_df)
    
    if len(combined_df) < 50:
        print(f"\n❌ Insufficient samples: {len(combined_df)} (need at least 50)")
        return None
    
    primary_ticker = loader.tickers[0]
    print(f"  Creating target labels using {primary_ticker}...")
    
    if use_significant_moves:
        print("  Mode: Predicting SIGNIFICANT moves (>2%)")
        combined_df = engineer.create_target_label_significant(combined_df, target_ticker=primary_ticker, threshold=0.02)
    else:
        print("  Mode: Predicting ANY move (UP/DOWN)")
        combined_df = engineer.create_target_label(combined_df, target_ticker=primary_ticker, forward_days=1)
    
    output_file = f'data/{category.lower()}_data.csv'
    combined_df.to_csv(output_file)
    print(f"  ✓ Saved: {combined_df.shape[0]} samples, {combined_df.shape[1]-1} features (includes {len([c for c in combined_df.columns if c.startswith('macro_')])} macro features)")
    
    print("\n[3/6] Training XGBoost Model...")
    use_feature_selection = combined_df.shape[1] > 100
    top_k = min(150, combined_df.shape[1] - 2)  # Select top 150 or all if less
    
    classifier = XGBoostMarketClassifier(
        random_state=42, 
        use_feature_selection=use_feature_selection,
        top_k_features=top_k
    )
    X_train, X_test, y_train, y_test = classifier.prepare_data(combined_df)
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Features used: {len(classifier.feature_names)}")
    
    if len(X_train) < 30:
        print("\n⚠️  Warning: Small training set. Results may not be reliable.")
    
    use_tuning = len(X_train) > 100
    classifier.train(X_train, y_train, use_tuning=use_tuning)
    
    print("\n[4/6] Evaluating Model...")
    metrics = classifier.evaluate(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print(f"MODEL PERFORMANCE - {loader.description}")
    print("="*70)
    print(f"Training Accuracy:   {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy:       {metrics['test_accuracy']:.4f}")
    print(f"Test F1-Score:       {metrics['test_f1']:.4f}")
    print(f"Test ROC-AUC:        {metrics['test_roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    print("\n[5/6] Generating Visualizations...")
    result_prefix = f'results/{category.lower()}'
    classifier.plot_confusion_matrix(save_path=f'{result_prefix}_confusion_matrix.png')
    classifier.plot_roc_curve(X_test, y_test, save_path=f'{result_prefix}_roc_curve.png')
    importance_df = classifier.plot_feature_importance(top_n=20, save_path=f'{result_prefix}_feature_importance.png')
    
    print("  Generating SHAP explanations (model interpretability)...")
    classifier.plot_shap_explanations(X_test, y_test, top_n=15, 
                                     save_path=f'{result_prefix}_shap_explanations.png')
    
    print("  Macro overlay ready; VIX + Treasury features embedded in every epoch (no simulations run).")
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    combined_features = np.vstack([X_train, X_test])
    combined_targets = np.concatenate([y_train, y_test])
    if len(combined_targets) > 80:
        try:
            tscv = TimeSeriesSplit(n_splits=4 if len(combined_targets) > 150 else 3)
            cv_model = xgb.XGBClassifier(**classifier.model.get_params())
            cv_scores = cross_val_score(cv_model, combined_features, combined_targets, cv=tscv, scoring='roc_auc', n_jobs=-1)
            print(f"\n[Cross-validation] Rolling ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        except Exception as exc:
            print(f"  ⚠️  TimeSeries CV failed: {exc}")

    print("\n[6/6] Making Next-Day Prediction...")
    exclude_cols = ['target', 'future_close'] + (['return'] if use_significant_moves else [])
    latest_features = combined_df.drop(columns=[col for col in exclude_cols if col in combined_df.columns]).iloc[[-1]]
    
    try:
        import yfinance as yf
        stock = yf.Ticker(primary_ticker)
        hist = stock.history(period='1d')
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else None
    except:
        current_price = None
    
    next_day_pred = classifier.predict_next_day(latest_features, current_price=current_price)
    
    print("\n" + "="*70)
    print(f"NEXT DAY PREDICTION - {primary_ticker}")
    print("="*70)
    print(f"Predicted Movement:  {next_day_pred['prediction']}")
    print(f"Confidence:          {next_day_pred['confidence']:.2%}")
    print(f"P(DOWN):             {next_day_pred['probability_down']:.2%}")
    print(f"P(UP):               {next_day_pred['probability_up']:.2%}")
    
    model_file = f'models/{category.lower()}_xgboost_model.json'
    classifier.save_model(model_file)
    
    metrics_file = f'results/{category.lower()}_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"MARKET MOVEMENT CLASSIFIER - {loader.description}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Category: {category}\n")
        f.write(f"Tickers: {', '.join(loader.tickers)}\n")
        f.write(f"SDG Aligned: {'Yes' if loader.sdg_aligned else 'No'}\n")
        if sdg_info.get('sdg_aligned'):
            f.write(f"SDG #{sdg_info['sdg_number']}: {sdg_info['sdg_name']}\n")
        f.write("\n")
        f.write(f"Training Accuracy:   {metrics['train_accuracy']:.4f}\n")
        f.write(f"Test Accuracy:       {metrics['test_accuracy']:.4f}\n")
        f.write(f"Test F1-Score:       {metrics['test_f1']:.4f}\n")
        f.write(f"Test ROC-AUC:        {metrics['test_roc_auc']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']) + "\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'] + "\n\n")
        f.write("Next Day Prediction:\n")
        f.write(f"Ticker: {primary_ticker}\n")
        f.write(f"Movement: {next_day_pred['prediction']}\n")
        f.write(f"Confidence: {next_day_pred['confidence']:.2%}\n")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"✓ Data saved to: {output_file}")
    print(f"✓ Model saved to: {model_file}")
    print(f"✓ Metrics saved to: {metrics_file}")
    print(f"✓ Visualizations in: results/")
    
    return {
        'loader': loader,
        'classifier': classifier,
        'metrics': metrics,
        'prediction': next_day_pred,
        'combined_df': combined_df,
        'primary_ticker': primary_ticker,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'raw_data': raw_data
    }


def train_and_predict(tickers, category='CUSTOM', use_significant_moves=False, train_per_ticker=True):
    if train_per_ticker and len(tickers) > 1:
        print(f"Training separate models for {len(tickers)} tickers...")
        
        market_data = None
        try:
            import yfinance as yf
            spy_data = yf.download('SPY', period='5y', interval='1d', progress=False, auto_adjust=True)
            if not spy_data.empty:
                spy_data.columns = [f'SPY_{col}' for col in spy_data.columns]
                market_data = spy_data
        except:
            pass
        
        results = {}
        for ticker in tickers:
            print(f"\n{'='*70}")
            print(f"Training model for {ticker}")
            print(f"{'='*70}")
            result = train_single_ticker(ticker, market_data=market_data, use_significant_moves=use_significant_moves)
            if result:
                results[ticker] = result
        
        if not results:
            return None
        
        return {
            'results': results,
            'tickers': list(results.keys()),
            'train_per_ticker': True
        }
    else:
        return main(category=category, custom_tickers=tickers, use_significant_moves=use_significant_moves)


def run_multiple_categories(categories=None):
    if categories is None:
        categories = ['SDG_CLEAN_ENERGY', 'TECH', 'FINANCE']
    
    print("\n" + "="*70)
    print("RUNNING MULTI-CATEGORY ANALYSIS")
    print("="*70)
    
    results = {}
    
    for category in categories:
        print(f"\n\n{'='*70}")
        print(f"CATEGORY: {category}")
        print(f"{'='*70}\n")
        
        try:
            result = main(category=category)
            if result:
                results[category] = result
        except Exception as e:
            print(f"\n❌ Failed for {category}: {e}")
            continue
    
    if results:
        print("\n\n" + "="*70)
        print("CATEGORY COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Category':<25} {'Test Acc':<12} {'ROC-AUC':<12} {'SDG':>10}")
        print("-"*70)
        
        for cat, res in results.items():
            metrics = res['metrics']
            sdg = "✓ Yes" if res['loader'].sdg_aligned else "  No"
            print(f"{cat:<25} {metrics['test_accuracy']:>10.2%}  {metrics['test_roc_auc']:>10.3f}  {sdg:>10}")
        
        print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Movement Classifier')
    parser.add_argument('--category', type=str, default='SDG_CLEAN_ENERGY',
                      help='Category to analyze (or use --list to see options)')
    parser.add_argument('--list', action='store_true',
                      help='List all available categories')
    parser.add_argument('--multi', action='store_true',
                      help='Run on multiple categories')
    parser.add_argument('--custom', type=str, nargs='+',
                      help='Custom ticker symbols (e.g., --custom TSLA NFLX AMZN)')
    parser.add_argument('--significant', action='store_true',
                      help='Predict only significant moves (>2%%)')
    
    args = parser.parse_args()
    
    # List categories
    if args.list:
        CleanEnergyDataLoader.list_categories()
    # Multi-category analysis
    elif args.multi:
        run_multiple_categories(['SDG_CLEAN_ENERGY', 'SDG_HEALTH', 'TECH', 'FINANCE', 'CONSUMER'])
    # Custom tickers
    elif args.custom:
        print(f"Running with custom tickers: {args.custom}")
        main(category='CUSTOM', custom_tickers=args.custom, use_significant_moves=args.significant)
    # Single category
    else:
        main(category=args.category, use_significant_moves=args.significant)