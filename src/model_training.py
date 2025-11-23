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
        
        print("Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            verbose=False
        )
        print("Training complete!")
        
        return self.model
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
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
        print(f"Confusion matrix saved to {save_path}")
    
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
        print(f"ROC curve saved to {save_path}")
    
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
        print(f"Feature importance plot saved to {save_path}")
        
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
    
    def save_model(self, path='models/xgboost_model.json'):
        """Save trained model"""
        self.model.save_model(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/xgboost_model.json'):
        """Load trained model"""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        print(f"Model loaded from {path}")