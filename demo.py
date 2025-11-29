#!/usr/bin/env python3
"""
Quick Demo Script for Hackathon Judges
Demonstrates the Market Movement Classifier with SDG-aligned stocks
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import main

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def demo():
    """Run a quick demo for hackathon judges"""
    
    print_header("🚀 MARKET MOVEMENT CLASSIFIER - HACKATHON DEMO")
    
    print("""
This demo showcases:
  ✓ SDG-aligned stock prediction (Clean Energy - SDG #7)
  ✓ Advanced feature engineering (200+ features)
  ✓ XGBoost with hyperparameter tuning
  ✓ SHAP explainability (model interpretability)
  ✓ Backtesting framework (historical performance)
  ✓ Market regime indicators (SPY correlation)
  ✓ Next-day UP/DOWN prediction
  
Problem Statement Alignment:
  ✓ Binary classification (UP/DOWN)
  ✓ Next-day prediction (1-day forward)
  ✓ Historical time-series data (2 years)
  ✓ XGBoost + Feature Engineering
  ✓ SDG Alignment (+20% bonus)
  
Key Innovations:
  ✓ Model Interpretability (SHAP values)
  ✓ Backtesting with realistic trading simulation
  ✓ Comprehensive feature engineering
  ✓ Production-ready web application
    """)
    
    input("Press ENTER to start the demo...")
    
    print_header("Running SDG Clean Energy Prediction")
    print("Category: SDG #7 - Affordable and Clean Energy")
    print("Tickers: ICLN, TAN, ENPH, FSLR, RUN, SEDG")
    print("\nThis will take 60-90 seconds (includes SHAP & backtesting)...\n")
    
    start_time = time.time()
    
    # Run the main pipeline
    result = main(category='SDG_CLEAN_ENERGY')
    
    elapsed = time.time() - start_time
    
    if result:
        print_header("✅ DEMO COMPLETE!")
        
        metrics = result['metrics']
        prediction = result['prediction']
        
        # Enhanced metrics display
        precision = metrics.get('test_precision', 0)
        recall = metrics.get('test_recall', 0)
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    PERFORMANCE METRICS                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Test Accuracy:    {metrics['test_accuracy']:>6.2%}                                    ║
║  ROC-AUC Score:    {metrics['test_roc_auc']:>6.3f}                                    ║
║  F1-Score:         {metrics['test_f1']:>6.3f}                                    ║
║  Precision:        {precision:>6.3f}                                    ║
║  Recall:           {recall:>6.3f}                                    ║
╠══════════════════════════════════════════════════════════════════╣
║                    NEXT-DAY PREDICTION                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Movement:         {prediction['prediction']:>6s}                                    ║
║  Confidence:       {prediction['confidence']:>6.2%}                                    ║
║  P(UP):            {prediction['probability_up']:>6.2%}                                    ║
║  P(DOWN):          {prediction['probability_down']:>6.2%}                                    ║
╠══════════════════════════════════════════════════════════════════╣
║                    SDG IMPACT                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  ✓ Aligned with SDG #7: Affordable and Clean Energy             ║
║  ✓ Supports renewable energy investment decisions                ║
║  ✓ Enables better capital allocation to clean energy             ║
║  ✓ Model interpretability via SHAP values                       ║
║  ✓ Backtesting validates historical performance                 ║
╠══════════════════════════════════════════════════════════════════╣
║                    EXECUTION INFO                                ║
╠══════════════════════════════════════════════════════════════════╣
║  Execution Time:   {elapsed:>6.1f} seconds                                    ║
╚══════════════════════════════════════════════════════════════════╝

Files Generated:
  ✓ Model:           models/sdg_clean_energy_xgboost_model.json
  ✓ Data:            data/sdg_clean_energy_data.csv
  ✓ Visualizations: results/sdg_clean_energy_*.png
    - Confusion Matrix
    - ROC Curve
    - Feature Importance
    - SHAP Explanations (NEW!)
    - Backtest Results (NEW!)
  ✓ Metrics:         results/sdg_clean_energy_metrics.txt
        """)
        
        print("\n" + "="*70)
        print("📊 Check the 'results/' folder for all visualizations!")
        print("💡 SHAP explanations show WHY the model makes predictions")
        print("📈 Backtest results show historical trading performance")
        print("="*70 + "\n")
    else:
        print("\n❌ Demo failed. Please check error messages above.\n")

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

