import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve, f1_score)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

class XGBoostMarketClassifier:
    def __init__(self, random_state=42, use_feature_selection=True, top_k_features=100):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=top_k_features) if use_feature_selection else None
        self.feature_names = None
        self.selected_features = None
        self.metrics = {}
        self.training_data_ = None
        self.validation_data_ = None
        
    def prepare_data(self, df, target_col='target', test_size=0.2):
        exclude_cols = [target_col, 'future_close', 'return'] + [col for col in df.columns if 'Date' in col or 'date' in col.lower()]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        print(f"  Cleaning data...")
        print(f"    Before: {X.shape}, NaN: {X.isna().sum().sum()}, Inf: {np.isinf(X).sum().sum()}")
        
        X = X.replace([np.inf, -np.inf], np.nan)
        
        X = X.ffill().bfill().fillna(0)
        
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                col_std = X[col].std()
                col_mean = X[col].mean()
                
                if col_std > 0 and not np.isnan(col_std) and not np.isinf(col_std):
                    lower_bound = col_mean - 5 * col_std
                    upper_bound = col_mean + 5 * col_std
                    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        print(f"    After: {X.shape}, NaN: {X.isna().sum().sum()}, Inf: {np.isinf(X).sum().sum()}")
        
        self.feature_names = X.columns.tolist()
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target distribution: UP={sum(y==1)}, DOWN={sum(y==0)}")
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        assert not np.any(np.isinf(X_train.values)), "X_train contains inf before scaling"
        assert not np.any(np.isinf(X_test.values)), "X_test contains inf before scaling"
        
        if self.feature_selector is not None and len(self.feature_names) > self.feature_selector.k:
            print(f"  Selecting top {self.feature_selector.k} features from {len(self.feature_names)}...")
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            X_test_selected = self.feature_selector.transform(X_test)
            
            selected_mask = self.feature_selector.get_support()
            self.selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_mask[i]]
            self.feature_names = self.selected_features
            
            print(f"  Selected {len(self.feature_names)} features")
            X_train, X_test = X_train_selected, X_test_selected
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, params=None, use_tuning=True):
        if use_tuning and len(X_train) > 100:
            print("  Training with hyperparameter tuning...")
            self.training_data_ = (X_train, y_train)
            self.validation_data_ = None
            return self.train_with_tuning(X_train, y_train, n_iter=30)
        
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.08,
                'n_estimators': 200,
                'objective': 'binary:logistic',
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'min_child_weight': 3,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'gamma': 0.15,
                'reg_alpha': 0.2,
                'reg_lambda': 1.2,
                'scale_pos_weight': 1.0  # Adjust if class imbalance
            }
        
        self.model = xgb.XGBClassifier(**params)
        
        validation_fraction = 0.15 if len(X_train) >= 80 else 0.1
        val_size = max(20, int(len(X_train) * validation_fraction))
        if val_size >= len(X_train):
            val_size = max(1, len(X_train) // 5)
        train_size = len(X_train) - val_size
        if train_size <= 0:
            train_size = len(X_train) - 1
            val_size = 1
        
        X_fit, y_fit = X_train[:train_size], y_train[:train_size]
        X_val, y_val = X_train[train_size:], y_train[train_size:]
        self.training_data_ = (X_fit, y_fit)
        self.validation_data_ = (X_val, y_val) if len(X_val) else None
        
        print("  Training XGBoost model with early stopping...")
        eval_set = [(X_fit, y_fit)]
        fit_params = {'verbose': False}
        if len(X_val):
            eval_set.append((X_val, y_val))
            fit_params.update({
                'eval_set': eval_set,
                'early_stopping_rounds': 30
            })
        else:
            fit_params['eval_set'] = eval_set
        
        self.model.fit(X_fit, y_fit, **fit_params)
        print("  ✓ Training complete!")
        
        return self.model
    
    def train_with_tuning(self, X_train, y_train, n_iter=30):
        from sklearn.model_selection import RandomizedSearchCV
        
        param_distributions = {
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.05, 0.08, 0.1, 0.12, 0.15],
            'n_estimators': [150, 200, 250, 300],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.75, 0.8, 0.85, 0.9],
            'colsample_bytree': [0.75, 0.8, 0.85, 0.9],
            'gamma': [0, 0.1, 0.15, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.2, 0.3],
            'reg_lambda': [0.5, 1.0, 1.2, 1.5, 2.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        print(f"  Tuning hyperparameters ({n_iter} iterations with TimeSeriesSplit)...")
        tscv = TimeSeriesSplit(n_splits=3)
        
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=tscv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        self.model = random_search.best_estimator_
        self.validation_data_ = None
        print(f"  ✓ Best CV ROC-AUC: {random_search.best_score_:.4f}")
        print(f"  Best params: {random_search.best_params_}")
        
        return self.model
    
    def train_ensemble(self, X_train, y_train):
        print("  Training ensemble (XGB + LightGBM + RF)...")
        
        xgb_model = xgb.XGBClassifier(
            max_depth=5, learning_rate=0.1, n_estimators=150,
            random_state=self.random_state, use_label_encoder=False
        )
        
        lgb_model = lgb.LGBMClassifier(
            max_depth=5, learning_rate=0.1, n_estimators=150,
            random_state=self.random_state, verbose=-1
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=7, random_state=self.random_state
        )
        
        self.model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ],
            voting='soft',
            weights=[2, 2, 1]
        )
        
        self.model.fit(X_train, y_train)
        print("  ✓ Ensemble trained!")
        
        return self.model
    
    def time_series_cv(self, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled = scaler.transform(X_test_fold)
            
            self.model.fit(X_train_scaled, y_train_fold)
            score = self.model.score(X_test_scaled, y_test_fold)
            cv_scores.append(score)
            
            print(f"    Fold {fold}: {score:.3f}")
        
        print(f"  Mean CV: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
        
        return cv_scores
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        train_features = X_train
        train_target = y_train
        if self.training_data_ is not None:
            train_features, train_target = self.training_data_
        y_train_pred = self.model.predict(train_features)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score
        val_accuracy = None
        if self.validation_data_ is not None:
            X_val, y_val = self.validation_data_
            if len(X_val):
                val_preds = self.model.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_preds)
        
        self.metrics = {
            'train_accuracy': accuracy_score(train_target, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'classification_report': classification_report(y_test, y_test_pred)
        }
        if val_accuracy is not None:
            self.metrics['val_accuracy'] = val_accuracy
        
        return self.metrics
    
    def plot_confusion_matrix(self, save_path='results/confusion_matrix.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
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
        print(f"  ✓ Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, X_test, y_test, save_path='results/roc_curve.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
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
        print(f"  ✓ ROC curve saved to {save_path}")
    
    def plot_feature_importance(self, top_n=20, save_path='results/feature_importance.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        else:
            print("  ⚠️  Cannot extract feature importances")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Feature importance plot saved to {save_path}")
        
        return importance_df
    
    def plot_shap_explanations(self, X_test, y_test, top_n=15, save_path='results/shap_explanations.png'):
        try:
            import shap
        except ImportError:
            print("  ⚠️  SHAP not installed. Install with: pip install shap")
            return None
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            sample_size = min(100, len(X_test))
            X_sample = X_test[:sample_size]
            
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            feature_names = self.feature_names[:len(shap_values[0])] if isinstance(shap_values, list) else self.feature_names
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                            max_display=top_n, show=False)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ SHAP explanations saved to {save_path}")
            
            bar_path = save_path.replace('.png', '_bar.png')
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                            plot_type="bar", max_display=top_n, show=False)
            plt.tight_layout()
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ SHAP bar plot saved to {bar_path}")
            
            return shap_values
            
        except Exception as e:
            print(f"  ⚠️  SHAP plot failed: {e}")
            return None
    
    def predict_next_day(self, latest_features, current_price=None):
        latest_features = latest_features.replace([np.inf, -np.inf], np.nan)
        latest_features = latest_features.ffill().bfill().fillna(0)
        
        if self.selected_features is not None:
            available_features = [f for f in self.selected_features if f in latest_features.columns]
            if len(available_features) != len(self.selected_features):
                missing = set(self.selected_features) - set(available_features)
                for feat in missing:
                    latest_features[feat] = 0
            latest_features = latest_features[self.selected_features]
        else:
            available_features = [f for f in self.feature_names if f in latest_features.columns]
            latest_features = latest_features[available_features]
            missing = set(self.feature_names) - set(available_features)
            for feat in missing:
                latest_features[feat] = 0
            latest_features = latest_features[self.feature_names]
        
        latest_scaled = self.scaler.transform(latest_features)
        prediction = self.model.predict(latest_scaled)[0]
        probability = self.model.predict_proba(latest_scaled)[0]
        
        direction = 'UP' if prediction == 1 else 'DOWN'
        confidence = max(probability)
        
        estimated_price = None
        if current_price is not None and current_price > 0:
            if direction == 'UP':
                expected_return = 0.01 + (0.01 * confidence)
            else:
                expected_return = -0.01 - (0.01 * confidence)
            
            estimated_price = current_price * (1 + expected_return)
        
        return {
            'prediction': direction,
            'probability_down': probability[0],
            'probability_up': probability[1],
            'confidence': confidence,
            'estimated_price': estimated_price,
            'arrow': '↑' if direction == 'UP' else '↓'
        }
    
    def save_model(self, path='models/xgboost_model.json'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"  ✓ Model saved to {path}")
    
    def load_model(self, path='models/xgboost_model.json'):
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        print(f"  ✓ Model loaded from {path}")