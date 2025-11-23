import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve, f1_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib

class XGBoostMarketClassifier:
    """XGBoost classifier for market movement prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
    def prepare_data(self, df, target_col='target', test_size=0.2):
        """Prepare train/test split with proper scaling"""
        # Remove non-feature columns
        exclude_cols = [target_col, 'future_close'] + [col for col in df.columns if 'Date' in col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data (time-series aware: no shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n📊 Data Split:")
        print(f"   • Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"   • Testing samples: {len(X_test)} ({test_size*100:.0f}%)")
        print(f"   • Total features: {len(self.feature_names)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, params=None):
        """Train XGBoost classifier"""
        if params is None:
            params = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1
            }
        
        self.model = xgb.XGBClassifier(**params)
        
        print("\n🚀 Training XGBoost model...")
        print(f"   Parameters: {params}")
        
        self.model.fit(
            X_train, y_train,
            verbose=False
        )
        print("✓ Training complete!")
        
        return self.model
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        print("\n📈 Evaluating model performance...")
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        self.metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'classification_report': classification_report(y_test, y_test_pred)
        }
        
        # Display metrics
        print("\n" + "="*60)
        print("   MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"\n📊 Accuracy:")
        print(f"   • Training: {self.metrics['train_accuracy']:.4f} ({self.metrics['train_accuracy']*100:.2f}%)")
        print(f"   • Testing:  {self.metrics['test_accuracy']:.4f} ({self.metrics['test_accuracy']*100:.2f}%)")
        
        print(f"\n🎯 Additional Metrics:")
        print(f"   • F1 Score:  {self.metrics['test_f1']:.4f}")
        print(f"   • ROC AUC:   {self.metrics['test_roc_auc']:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(self.metrics['classification_report'])
        
        print(f"🔢 Confusion Matrix:")
        cm = self.metrics['confusion_matrix']
        print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
        return self.metrics
    
    def plot_confusion_matrix(self, save_path='results/confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = self.metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
        plt.title('Confusion Matrix - Market Movement Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, X_test, y_test, save_path='results/roc_curve.png'):
        """Plot ROC curve"""
        y_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.metrics["test_roc_auc"]:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Market Movement Classifier')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ ROC curve saved to {save_path}")
    
    def plot_feature_importance(self, top_n=20, save_path='results/feature_importance.png'):
        """Plot top feature importances"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Feature importance plot saved to {save_path}")
        
        return importance_df
    
    def predict_next_day(self, latest_features):
        """Predict next day's market movement"""
        latest_scaled = self.scaler.transform(latest_features)
        prediction = self.model.predict(latest_scaled)[0]
        probability = self.model.predict_proba(latest_scaled)[0]
        
        return {
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'probability_down': probability[0],
            'probability_up': probability[1],
            'confidence': max(probability)
        }
    
    def save_model(self, model_path='models/xgboost_model.json', 
                   scaler_path='models/scaler.pkl'):
        """Save trained model and scaler"""
        self.model.save_model(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"   ✓ Model saved to {model_path}")
        print(f"   ✓ Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='models/xgboost_model.json',
                   scaler_path='models/scaler.pkl'):
        """Load trained model and scaler"""
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"   ✓ Model loaded from {model_path}")
        print(f"   ✓ Scaler loaded from {scaler_path}")


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("   MODEL TRAINING - UNIVERSAL STOCK PREDICTOR")
    print("="*60)
    
    # Load configuration
    config_path = 'data/config.json'
    if not os.path.exists(config_path):
        print("\n❌ Error: config.json not found!")
        print("Please run the data loader and feature engineering scripts first.")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    target_ticker = config['target_ticker']
    forward_days = config.get('forward_days', 1)
    
    print(f"\n📊 Configuration:")
    print(f"   • Target ticker: {target_ticker}")
    print(f"   • Prediction horizon: {forward_days} day(s)")
    
    # Load processed features
    data_path = 'data/processed_features.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ Error: {data_path} not found!")
        print("Please run the feature engineering script first.")
        return None
    
    print(f"\n⏳ Loading processed features...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Initialize classifier
    classifier = XGBoostMarketClassifier(random_state=42)
    
    # Prepare data
    test_size = 0.2
    X_train, X_test, y_train, y_test = classifier.prepare_data(df, test_size=test_size)
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Evaluate model
    metrics = classifier.evaluate(X_train, X_test, y_train, y_test)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    classifier.plot_confusion_matrix()
    classifier.plot_roc_curve(X_test, y_test)
    top_features = classifier.plot_feature_importance(top_n=20)
    
    # Save model
    print("\n💾 Saving model...")
    classifier.save_model()
    
    # Test prediction on latest data
    print("\n🔮 Testing prediction on latest data...")
    latest_features = df.drop(['target', 'future_close'], axis=1).iloc[[-1]]
    prediction = classifier.predict_next_day(latest_features)
    
    print(f"\n{'='*60}")
    print(f"   PREDICTION FOR {target_ticker}")
    print(f"{'='*60}")
    print(f"\n🎯 Next {forward_days} day(s) prediction: {prediction['prediction']}")
    print(f"   • Probability UP:   {prediction['probability_up']:.2%}")
    print(f"   • Probability DOWN: {prediction['probability_down']:.2%}")
    print(f"   • Confidence:       {prediction['confidence']:.2%}")
    
    # Save final metrics to config
    config['model_training_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    config['test_accuracy'] = float(metrics['test_accuracy'])
    config['test_f1'] = float(metrics['test_f1'])
    config['test_roc_auc'] = float(metrics['test_roc_auc'])
    config['latest_prediction'] = prediction
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print(f"   • Model accuracy: {metrics['test_accuracy']:.2%}")
    print(f"   • All results saved to: results/")
    print(f"   • Model saved to: models/")
    print("="*60)
    
    return classifier, metrics


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run training
    model, metrics = main()