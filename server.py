from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import sys
import os
import pandas as pd
import numpy as np
import warnings
import json
from datetime import datetime
import threading
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import zipfile

# Import from main.py instead of duplicating logic
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

try:
    from main import train_and_predict
except ImportError as e:
    print(f"ERROR: Failed to import from main.py: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)
CORS(app)

models_cache = {}
predictions_cache = {}
cache_lock = threading.Lock()

CATEGORIES = {
    'SDG_CLEAN_ENERGY': ['ICLN', 'TAN', 'ENPH', 'FSLR'],
    'SDG_HEALTH': ['JNJ', 'PFE', 'UNH', 'ABBV'],
    'TECH': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
    'FINANCE': ['JPM', 'BAC', 'GS', 'MS'],
    'CONSUMER': ['AMZN', 'WMT', 'HD', 'NKE'],
    'ENERGY_TRADITIONAL': ['XOM', 'CVX', 'COP', 'SLB']
}

def get_current_price(ticker):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return None

def get_currency_symbol(ticker):
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        return '₹'
    elif ticker.endswith('.L') or ticker.endswith('.TO'):
        return '£'
    elif ticker.endswith('.T') or ticker.endswith('.HK'):
        return '¥'
    else:
        return '$'

def get_fundamentals(ticker):
    pe_ratio = None
    dividend_yield = None
    next_earnings = None
    last_earnings = None
    eps = None
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        fast_info = getattr(stock, "fast_info", None)
        if fast_info:
            pe_ratio = fast_info.get("trailing_pe") or fast_info.get("forward_pe")
            dividend_yield = fast_info.get("dividend_yield")
            if dividend_yield is not None:
                dividend_yield = float(dividend_yield) * 100.0
        info = getattr(stock, "info", {}) or {}
        if not pe_ratio:
            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        if dividend_yield is None:
            dy = info.get("dividendYield")
            if dy is not None:
                dividend_yield = float(dy) * 100.0
        try:
            cal = getattr(stock, "calendar", None)
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                if "Earnings Date" in cal.index:
                    dates = cal.loc["Earnings Date"].values
                    if len(dates) > 0:
                        next_earnings = pd.to_datetime(dates[0]).isoformat()
        except Exception:
            pass
        try:
            earnings_dates = getattr(stock, "earnings_dates", None)
            if callable(earnings_dates):
                ed_df = earnings_dates(limit=4)
            else:
                ed_df = earnings_dates
            if isinstance(ed_df, pd.DataFrame) and not ed_df.empty:
                last_row = ed_df.sort_index().iloc[-1]
                last_earnings = pd.to_datetime(last_row.name).isoformat()
                eps = float(last_row.get("EPS Actual")) if "EPS Actual" in last_row else None
        except Exception:
            pass
    except Exception:
        pass
    return {
        "peRatio": float(pe_ratio) if pe_ratio not in (None, np.nan) else None,
        "dividendYield": float(dividend_yield) if dividend_yield not in (None, np.nan) else None,
        "nextEarningsDate": next_earnings,
        "lastEarningsDate": last_earnings,
        "eps": eps
    }

def normalize_ticker_input(tickers_input):
    if isinstance(tickers_input, str):
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    else:
        tickers = [str(t).strip().upper() for t in tickers_input if str(t).strip()]
    return tickers

def compute_risk_level(ticker_df):
    if ticker_df is None:
        return 'Unknown', None
    candidate_cols = [col for col in ticker_df.columns if 'return_1d' in col.lower()] or \
                     [col for col in ticker_df.columns if col.lower().startswith('return')]
    volatility = None
    for col in candidate_cols:
        series = ticker_df[col].dropna()
        if len(series) > 25:
            volatility = series.rolling(20).std().iloc[-1]
            break
    if volatility is None or np.isnan(volatility):
        return 'Unknown', None
    vol_pct = float(volatility) * 100
    if vol_pct < 1:
        return 'Low', vol_pct
    if vol_pct < 2:
        return 'Medium', vol_pct
    return 'High', vol_pct

def derive_investment_opinion(prediction_payload, ticker_df=None, current_price=None):
    direction = prediction_payload.get('prediction')
    confidence = prediction_payload.get('confidence', 0)
    prob_up = prediction_payload.get('probability_up', 0)
    prob_down = prediction_payload.get('probability_down', 0)
    estimated_price = prediction_payload.get('estimated_price')
    
    action = 'HOLD'
    rationale = "Momentum signals are mixed; maintain current exposure."
    
    if direction == 'UP':
        if confidence >= 0.65:
            action = 'BUY'
            rationale = f"Model detects {prob_up*100:.1f}% probability of upside with solid momentum."
        elif confidence >= 0.55:
            action = 'HOLD'
            rationale = f"Upside bias ({prob_up*100:.1f}% probability) but conviction is moderate."
    elif direction == 'DOWN':
        if confidence >= 0.65:
            action = 'SELL'
            rationale = f"Downside pressure dominates with {prob_down*100:.1f}% probability; consider trimming exposure."
        else:
            action = 'HOLD'
            rationale = f"Downside skew present but conviction is limited ({prob_down*100:.1f}% probability)."
    
    expected_move_pct = None
    if estimated_price and current_price and current_price > 0:
        expected_move_pct = ((estimated_price - current_price) / current_price) * 100
    elif direction in ('UP', 'DOWN'):
        bias = prob_up - prob_down
        expected_move_pct = bias * 5  # heuristic scale
    
    risk_level, volatility_pct = compute_risk_level(ticker_df)
    
    supporting_note = f"Risk level: {risk_level}"
    if volatility_pct is not None:
        supporting_note += f" (σ≈{volatility_pct:.2f}%)"
    
    return {
        'recommendation': action,
        'rationale': rationale,
        'expectedMovePct': round(expected_move_pct, 2) if expected_move_pct is not None else None,
        'riskLevel': risk_level,
        'riskNote': supporting_note
    }

def build_prediction_entry(ticker, result, sdg_alignment=None):
    pred = result['prediction']
    metrics = result['metrics']
    ticker_df = result.get('ticker_df')
    if ticker_df is None:
        ticker_df = result.get('combined_df')
    current_price = result.get('current_price') or get_current_price(ticker) or 0.0
    currency = get_currency_symbol(ticker)
    fundamentals = get_fundamentals(ticker)
    
    enriched = derive_investment_opinion(pred, ticker_df=ticker_df, current_price=current_price)
    estimated_price = pred.get('estimated_price', current_price)
    
    entry = {
        'ticker': ticker,
        'currentPrice': round(current_price, 2) if current_price else None,
        'estimatedPrice': round(estimated_price, 2) if estimated_price else None,
        'direction': pred.get('prediction'),
        'arrow': pred.get('arrow', '↑' if pred.get('prediction') == 'UP' else '↓'),
        'confidence': round(pred.get('confidence', 0) * 100, 1),
        'probabilityUp': round(pred.get('probability_up', 0) * 100, 1),
        'probabilityDown': round(pred.get('probability_down', 0) * 100, 1),
        'currency': currency,
        'isPositive': pred.get('prediction') == 'UP',
        'hasRealPrediction': True,
        'sdgAlignment': sdg_alignment or 'Not assessed',
        'peRatio': fundamentals.get('peRatio'),
        'dividendYield': fundamentals.get('dividendYield'),
        'nextEarningsDate': fundamentals.get('nextEarningsDate'),
        'lastEarningsDate': fundamentals.get('lastEarningsDate'),
        'eps': fundamentals.get('eps'),
        **enriched
    }
    
    expected_change = None
    if entry['currentPrice'] and entry['estimatedPrice']:
        try:
            expected_change = ((entry['estimatedPrice'] - entry['currentPrice']) / entry['currentPrice']) * 100
            entry['expectedChangePct'] = round(expected_change, 2)
        except ZeroDivisionError:
            entry['expectedChangePct'] = None
    else:
        entry['expectedChangePct'] = enriched.get('expectedMovePct')
    
    metric_entry = {
        'ticker': ticker,
        'accuracy': round(metrics.get('test_accuracy', 0), 4),
        'roc_auc': round(metrics.get('test_roc_auc', 0), 4)
    }
    if 'val_accuracy' in metrics:
        metric_entry['val_accuracy'] = round(metrics['val_accuracy'], 4)
    
    return entry, metric_entry

def prepare_prediction_payload(tickers):
    pipeline_result = get_or_create_model(tickers)
    if not pipeline_result:
        return None
    
    predictions = []
    metrics_list = []
    
    if pipeline_result.get('train_per_ticker') and 'results' in pipeline_result:
        results_map = pipeline_result['results']
        for ticker in tickers:
            result = results_map.get(ticker)
            if not result:
                continue
            entry, metric_entry = build_prediction_entry(ticker, result)
            predictions.append(entry)
            metrics_list.append(metric_entry)
    else:
        primary_ticker = pipeline_result.get('primary_ticker', tickers[0])
        loader = pipeline_result.get('loader')
        sdg_alignment = None
        if loader and hasattr(loader, 'sdg_aligned'):
            sdg_alignment = 'Yes' if getattr(loader, 'sdg_aligned') else 'No'
        entry, metric_entry = build_prediction_entry(primary_ticker, pipeline_result,
                                                     sdg_alignment=sdg_alignment)
        predictions.append(entry)
        metrics_list.append(metric_entry)
    
    if not predictions:
        return None
    
    avg_accuracy = sum(m.get('accuracy', 0) for m in metrics_list) / len(metrics_list)
    
    metadata = {
        'models_trained': len(predictions),
        'per_ticker_models': pipeline_result.get('train_per_ticker', False),
        'model_accuracy': avg_accuracy,
        'metrics': metrics_list,
        'requested_tickers': tickers,
        'primary_ticker': predictions[0]['ticker']
    }
    
    return {
        'predictions': predictions,
        'metadata': metadata,
        'pipeline_result': pipeline_result
    }

def build_visualization_context(ticker, result, combined=False):
    if combined:
        return {
            'classifier': result.get('classifier'),
            'X_test': result.get('X_test'),
            'y_test': result.get('y_test'),
            'tickers': [ticker],
            'loader': result.get('loader'),
            'raw_data': result.get('raw_data'),
            'ticker_df': result.get('combined_df'),
            'ticker': ticker,
            'primary_ticker': ticker,
            'combined_df': result.get('combined_df')
        }
    
    ticker_df = result.get('ticker_df')
    
    class MockLoader:
        def __init__(self, df, ticker_name):
            self.ticker_df = df
            self.ticker_name = ticker_name
            self.tickers = [ticker_name]
        
        def get_ticker_data(self, symbol):
            if symbol == self.ticker_name:
                return self.ticker_df
            return None
    
    mock_loader = MockLoader(ticker_df, ticker)
    
    return {
        'classifier': result.get('classifier'),
        'X_test': result.get('X_test'),
        'y_test': result.get('y_test'),
        'tickers': [ticker],
        'loader': mock_loader,
        'raw_data': result.get('raw_data'),
        'ticker_df': ticker_df,
        'ticker': ticker,
        'primary_ticker': ticker,
        'combined_df': ticker_df
    }

def build_visualization_bundle(ticker, result, metrics, combined=False, as_data_uri=True):
    context = build_visualization_context(ticker, result, combined=combined)
    return {
        'confusionMatrix': create_confusion_matrix_image(metrics['confusion_matrix'], as_data_uri=as_data_uri),
        'rocCurve': create_roc_curve_image(context, as_data_uri=as_data_uri),
        'featureImportance': create_feature_importance_image(context, as_data_uri=as_data_uri),
        'priceHistory': create_price_history_chart(context, as_data_uri=as_data_uri),
        'returnsDistribution': create_returns_distribution_chart(context, as_data_uri=as_data_uri),
        'metrics': {
            'train_accuracy': round(metrics.get('train_accuracy', 0), 4),
            'test_accuracy': round(metrics.get('test_accuracy', 0), 4),
            'test_f1': round(metrics.get('test_f1', 0), 4),
            'test_roc_auc': round(metrics.get('test_roc_auc', 0), 4),
            'val_accuracy': round(metrics.get('val_accuracy', 0), 4) if 'val_accuracy' in metrics else None,
            'confusion_matrix': metrics.get('confusion_matrix').tolist() if isinstance(metrics.get('confusion_matrix'), np.ndarray) else metrics.get('confusion_matrix')
        }
    }

def write_visualizations_to_zip(zip_file, ticker, viz_bundle):
    filename_map = {
        'confusionMatrix': 'confusion_matrix.png',
        'rocCurve': 'roc_curve.png',
        'featureImportance': 'feature_importance.png',
        'priceHistory': 'price_history.png',
        'returnsDistribution': 'returns_distribution.png'
    }
    for key, filename in filename_map.items():
        blob = viz_bundle.get(key)
        if not blob:
            continue
        zip_file.writestr(f'visualizations/{ticker}/{filename}', blob)
    # Save metrics JSON snapshot for convenience
    metrics_payload = viz_bundle.get('metrics')
    if metrics_payload:
        zip_file.writestr(f'visualizations/{ticker}/metrics.json', json.dumps(metrics_payload, indent=2))

def get_or_create_model(tickers):
    ticker_key = ','.join(sorted(tickers))
    
    with cache_lock:
        if ticker_key in models_cache:
            cache_time = models_cache[ticker_key].get('timestamp', 0)
            if time.time() - cache_time < 3600:
                return models_cache[ticker_key]
        
        print(f"Running main pipeline for: {ticker_key}")
        # Use the clean function from main.py - trains separate model for each ticker
        result = train_and_predict(tickers, category='CUSTOM', train_per_ticker=True)
        
        if result:
            result['timestamp'] = time.time()
            models_cache[ticker_key] = result
        
        return result

def create_confusion_matrix_image(cm, as_data_uri=True):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close()
    
    if as_data_uri:
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    return image_bytes

def create_roc_curve_image(pipeline_result, as_data_uri=True):
    from sklearn.metrics import roc_curve, auc
    
    classifier = pipeline_result['classifier']
    X_test = pipeline_result['X_test']
    y_test = pipeline_result['y_test']
    
    y_proba = classifier.model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2, color='#3b82f6')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close()
    
    if as_data_uri:
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    return image_bytes

def create_feature_importance_image(pipeline_result, top_n=15, as_data_uri=True):
    try:
        classifier = pipeline_result.get('classifier')
        if classifier is None:
            return None
        
        feature_names = classifier.feature_names
        
        if hasattr(classifier.model, 'feature_importances_'):
            importances = classifier.model.feature_importances_
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close()
        
        if as_data_uri:
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
        return image_bytes
    except Exception as e:
        print(f"  Warning: Feature importance chart failed: {e}")
        return None

def create_price_history_chart(pipeline_result, as_data_uri=True):
    try:
        tickers = pipeline_result.get('tickers', [])
        if not tickers:
            # Try to get from loader
            loader = pipeline_result.get('loader')
            if loader and hasattr(loader, 'tickers'):
                tickers = loader.tickers
        
        raw_data = pipeline_result.get('raw_data')
        loader = pipeline_result.get('loader')
        
        plt.figure(figsize=(10, 6))
        
        if raw_data is not None and tickers:
            for ticker in tickers:
                try:
                    if loader and hasattr(loader, 'get_ticker_data'):
                        ticker_df = loader.get_ticker_data(ticker)
                    else:
                        # Try to extract from raw_data directly
                        if isinstance(raw_data.columns, pd.MultiIndex):
                            ticker_df = raw_data[ticker].copy()
                        else:
                            cols = [col for col in raw_data.columns if ticker in str(col)]
                            ticker_df = raw_data[cols].copy()
                            if 'Close' not in ticker_df.columns and len(ticker_df.columns) > 0:
                                # Use first column as Close
                                ticker_df['Close'] = ticker_df.iloc[:, 0]
                    
                    if not ticker_df.empty and 'Close' in ticker_df.columns:
                        plt.plot(ticker_df.index, ticker_df['Close'], label=ticker, linewidth=2)
                except Exception as e:
                    print(f"  Warning: Could not plot {ticker}: {e}")
                    continue
        else:
            # Fallback: try to plot from ticker_df if available
            ticker_df = pipeline_result.get('ticker_df')
            if ticker_df is not None and 'Close' in ticker_df.columns:
                plt.plot(ticker_df.index, ticker_df['Close'], linewidth=2)
                ticker_name = pipeline_result.get('ticker', 'Stock')
                plt.title(f'{ticker_name} Price History (Last 2 Years)')
        
        if len(plt.gca().get_lines()) == 0:
            # No data plotted, create empty chart
            plt.text(0.5, 0.5, 'No price data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.title('Price History (Last 2 Years)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        if len(plt.gca().get_lines()) > 0:
            plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close()
        
        if as_data_uri:
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
        return image_bytes
    except Exception as e:
        print(f"  Warning: Price history chart failed: {e}")
        return None

def create_returns_distribution_chart(pipeline_result, as_data_uri=True):
    try:
        # Try to get data from different possible structures
        combined_df = pipeline_result.get('combined_df')
        ticker_df = pipeline_result.get('ticker_df')
        primary_ticker = pipeline_result.get('primary_ticker')
        ticker = pipeline_result.get('ticker')
        
        # Determine which dataframe and ticker to use
        if combined_df is not None and primary_ticker:
            df = combined_df
            ticker_name = primary_ticker
            return_col = f'{primary_ticker}_return_1d'
        elif ticker_df is not None:
            df = ticker_df
            ticker_name = ticker or 'Stock'
            return_col = 'return_1d'
        else:
            return None
        
        if return_col not in df.columns:
            return None
        
        returns = df[return_col].dropna()
        
        if len(returns) == 0:
            return None
        
        plt.figure(figsize=(8, 5))
        plt.hist(returns * 100, bins=50, color='#3b82f6', alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Return')
        plt.title(f'{ticker_name} Daily Returns Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close()
        
        if as_data_uri:
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
        return image_bytes
    except Exception as e:
        print(f"  Warning: Returns distribution chart failed: {e}")
        return None

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tickers = normalize_ticker_input(data.get('tickers', ''))
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        payload = prepare_prediction_payload(tickers)
        if not payload:
            return jsonify({'error': 'Failed to train models or insufficient data'}), 500
        
        return jsonify({
            'predictions': payload['predictions'],
            'metadata': payload['metadata']
        })
        
    except Exception as e:
        print(f"API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualizations', methods=['POST'])
def get_visualizations():
    try:
        data = request.get_json()
        tickers = normalize_ticker_input(data.get('tickers', ''))
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        pipeline_result = get_or_create_model(tickers)
        
        if not pipeline_result:
            return jsonify({'error': 'Model not found or failed to train'}), 500
        
        # Handle per-ticker results (new format)
        if pipeline_result.get('train_per_ticker') and 'results' in pipeline_result:
            primary_ticker = next((t for t in tickers if t in pipeline_result['results']), None)
            if not primary_ticker:
                return jsonify({'error': 'No results found for requested tickers'}), 500
            result = pipeline_result['results'][primary_ticker]
            metrics = result['metrics']
            visualizations = build_visualization_bundle(primary_ticker, result, metrics, combined=False, as_data_uri=True)
        else:
            # Handle legacy format (single combined model)
            metrics = pipeline_result.get('metrics', {})
            if not metrics:
                return jsonify({'error': 'No metrics found in pipeline result'}), 500
            primary_ticker = pipeline_result.get('primary_ticker', tickers[0])
            visualizations = build_visualization_bundle(primary_ticker, pipeline_result, metrics, combined=True, as_data_uri=True)
        
        return jsonify(visualizations)
        
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-report', methods=['POST'])
def download_report():
    try:
        data = request.get_json()
        tickers = normalize_ticker_input(data.get('tickers', ''))
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        payload = prepare_prediction_payload(tickers)
        if not payload:
            return jsonify({'error': 'Failed to generate report'}), 500
        
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            predictions_df = pd.DataFrame(payload['predictions'])
            zip_file.writestr('predictions.csv', predictions_df.to_csv(index=False))
            
            metrics_list = payload['metadata'].get('metrics') or []
            if metrics_list:
                metrics_df = pd.DataFrame(metrics_list)
                zip_file.writestr('metrics.csv', metrics_df.to_csv(index=False))
            
            summary_lines = [
                "Market Movement Classifier Report",
                f"Tickers: {', '.join(tickers)}",
                f"Generated: {datetime.utcnow().isoformat()}Z",
                f"Models trained: {payload['metadata']['models_trained']}",
                f"Average test accuracy: {payload['metadata']['model_accuracy']:.2%}",
                "",
                "Files:",
                "- predictions.csv: Enriched per-ticker predictions",
                "- metrics.csv: Test/validation metrics per ticker",
                "- visualizations/<ticker>/: PNG charts for each ticker",
                "",
                "Use cases:",
                "1. Share the zip with stakeholders",
                "2. Plug CSV into custom dashboards",
                "3. Inspect PNG charts for qualitative assessment"
            ]
            zip_file.writestr('README.txt', "\n".join(summary_lines))
            
            pipeline_result = payload['pipeline_result']
            if pipeline_result.get('train_per_ticker') and 'results' in pipeline_result:
                results_map = pipeline_result['results']
                for ticker in tickers:
                    if ticker not in results_map:
                        continue
                    result = results_map[ticker]
                    viz_bundle = build_visualization_bundle(ticker, result, result['metrics'], combined=False, as_data_uri=False)
                    write_visualizations_to_zip(zip_file, ticker, viz_bundle)
            else:
                ticker = payload['metadata'].get('primary_ticker', tickers[0])
                metrics = pipeline_result.get('metrics', {})
                if metrics:
                    viz_bundle = build_visualization_bundle(ticker, pipeline_result, metrics, combined=True, as_data_uri=False)
                    write_visualizations_to_zip(zip_file, ticker, viz_bundle)
        
        buffer.seek(0)
        filename = f"market-report-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.zip"
        return send_file(buffer, mimetype='application/zip', as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"Report download error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    categories = []
    for cat, tickers in CATEGORIES.items():
        categories.append({
            'name': cat,
            'tickers': tickers,
            'description': cat.replace('_', ' ').title()
        })
    return jsonify({'categories': categories})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'cached_models': len(models_cache)
    })

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    with cache_lock:
        models_cache.clear()
    return jsonify({'status': 'cache cleared'})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('/var/log/supervisor', exist_ok=True)
    
    print("="*70)
    print("MARKET MOVEMENT CLASSIFIER API SERVER")
    print("="*70)
    print("API Endpoints:")
    print("  POST /api/predict              - Predict custom tickers")
    print("  POST /api/visualizations       - Get model visualizations")
    print("  GET  /api/categories           - List all categories")
    print("  GET  /api/health               - Health check")
    print("  POST /api/clear-cache          - Clear model cache")
    print("="*70)
    print("\nServer starting on http://0.0.0.0:5000")
    print("Frontend should connect to: http://localhost:5000/api")
    print("="*70 + "\n")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"Fatal error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)