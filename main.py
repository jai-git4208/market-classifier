import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import CleanEnergyDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import XGBoostMarketClassifier

def main():
    """Main execution pipeline"""
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("="*70)
    print("CLEAN ENERGY STOCK MARKET MOVEMENT CLASSIFIER")
    print("SDG #7: Affordable & Clean Energy")
    print("="*70)
    
    # 1. DATA LOADING
    print("\n[1/6] Loading Data...")
    loader = CleanEnergyDataLoader(tickers=['ICLN', 'TAN', 'ENPH', 'FSLR'])
    raw_data = loader.download_data(period='2y', interval='1d')
    
    # 2. FEATURE ENGINEERING
    print("\n[2/6] Engineering Features...")
    engineer = FeatureEngineer()
    
    # Create features for each ticker
    all_ticker_features = []
    for ticker in loader.tickers:
        print(f"  Processing {ticker}...")
        ticker_df = loader.get_ticker_data(ticker)
        ticker_df = engineer.create_all_features(ticker_df, ticker_prefix='')
        ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
        all_ticker_features.append(ticker_df)
    
    # Combine all tickers
    combined_df = pd.concat(all_ticker_features, axis=1)
    combined_df = combined_df.dropna()
    
    # Create target label (using ICLN as primary ticker)
    print("  Creating target labels (NO DATA LEAKAGE)...")
    combined_df = engineer.create_target_label(combined_df, target_ticker='ICLN', forward_days=1)
    
    # Save processed data
    combined_df.to_csv('data/clean_energy_data.csv')
    print(f"  Saved processed data: {combined_df.shape[0]} samples, {combined_df.shape[1]} features")
    
    # 3. MODEL TRAINING
    print("\n[3/6] Training XGBoost Model...")
    classifier = XGBoostMarketClassifier(random_state=42)
    X_train, X_test, y_train, y_test = classifier.prepare_data(combined_df)
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    classifier.train(X_train, y_train)
    
    # 4. EVALUATION
    print("\n[4/6] Evaluating Model...")
    metrics = classifier.evaluate(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE METRICS")
    print("="*70)
    print(f"Training Accuracy:   {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy:       {metrics['test_accuracy']:.4f}")
    print(f"Test F1-Score:       {metrics['test_f1']:.4f}")
    print(f"Test ROC-AUC:        {metrics['test_roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # 5. VISUALIZATIONS
    print("\n[5/6] Generating Visualizations...")
    classifier.plot_confusion_matrix()
    classifier.plot_roc_curve(X_test, y_test)
    importance_df = classifier.plot_feature_importance(top_n=20)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # 6. NEXT-DAY PREDICTION
    print("\n[6/6] Making Next-Day Prediction...")
    latest_features = combined_df.drop(columns=['target', 'future_close']).iloc[[-1]]
    next_day_pred = classifier.predict_next_day(latest_features)
    
    print("\n" + "="*70)
    print("NEXT DAY PREDICTION")
    print("="*70)
    print(f"Predicted Movement:  {next_day_pred['prediction']}")
    print(f"Confidence:          {next_day_pred['confidence']:.2%}")
    print(f"P(DOWN):             {next_day_pred['probability_down']:.2%}")
    print(f"P(UP):               {next_day_pred['probability_up']:.2%}")
    
    # Save model
    classifier.save_model()
    
    # Save metrics to file
    with open('results/model_metrics.txt', 'w') as f:
        f.write("CLEAN ENERGY MARKET MOVEMENT CLASSIFIER\n")
        f.write("="*70 + "\n\n")
        f.write(f"Training Accuracy:   {metrics['train_accuracy']:.4f}\n")
        f.write(f"Test Accuracy:       {metrics['test_accuracy']:.4f}\n")
        f.write(f"Test F1-Score:       {metrics['test_f1']:.4f}\n")
        f.write(f"Test ROC-AUC:        {metrics['test_roc_auc']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']) + "\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'] + "\n\n")
        f.write("Next Day Prediction:\n")
        f.write(f"Movement: {next_day_pred['prediction']}\n")
        f.write(f"Confidence: {next_day_pred['confidence']:.2%}\n")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("✓ Data saved to: data/clean_energy_data.csv")
    print("✓ Model saved to: models/xgboost_model.json")
    print("✓ Results saved to: results/")
    
if __name__ == "__main__":
    main()