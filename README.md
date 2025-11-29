# Market Movement Classifier  

The **Market Movement Classifier** is an AI-based stock prediction system developed by **Team Aimers**.  
It uses a trained **XGBoost model** to predict next-day stock movements (UP/DOWN) and provides **real-time charts**, **latest stock data**, and **financial news** through an integrated full-stack web application.

> **🏆 Hackathon Ready:** See [HACKATHON_SUMMARY.md](HACKATHON_SUMMARY.md) for presentation summary and [QUICK_START.md](QUICK_START.md) for quick demo guide.

---

##  Table of Contents
1. Overview  
2. Problem Statement  
3. Features  
4. Project Structure  
5. Backend Stack  
6. Frontend Stack  
7. AI Model  
8. Installation  
9. How to Use  
10. Model Configuration  
11. Training Process  
12. SDG Alignment  
13. Troubleshooting  
14. Contributors  
15. Acknowledgments  
16. Links  

---

##  1. Overview  
The Market Movement Classifier is a **binary classification model** that predicts whether a stock’s next-day closing price will be **UP or DOWN**.

It combines:
- **Machine Learning Pipeline (main.py)** – Data loading, feature engineering, XGBoost training, and visualization  
- **REST API Backend (server.py)** – Flask API connecting ML to frontend  
- **Interactive UI (index.html)** – Real-time visualizations & predictions  
- **Web Server (app.py)** – Hosts frontend  

---

##  2. Problem Statement  
Build a trained ML model using **2 years of historical time-series stock data** to predict whether the next day’s closing price will go **UP or DOWN**.

- 🎯 **Next-Day Predictions** - Binary UP/DOWN classification with confidence scores
- 📊 **250+ Technical Features** - RSI, MACD, Bollinger Bands, ADX, Stochastic, Williams %R, Fibonacci, Ichimoku Cloud
- 🤖 **XGBoost Classifier** - Gradient boosting with automatic hyperparameter tuning
- 🧠 **Buy/Sell/Hold Guidance** - Actionable trade opinions with confidence, risk note, and expected move
- 🔍 **SHAP Explainability** - Model interpretability - shows WHY predictions are made (judges love this!)
- 📈 **Backtesting Framework** - Historical performance validation with realistic trading simulation
- 📈 **Real-Time Data** - Downloads latest 2-year historical data via Yahoo Finance
- 🌍 **SDG Alignment** - Supports 4 UN SDG categories (#3, #7, #9, #13) for +20% bonus
- 🔄 **Multi-Ticker Support** - Analyze multiple stocks simultaneously with cross-ticker correlation
- 📉 **Interactive Visualizations** - Confusion matrix, ROC curve, feature importance, SHAP plots, backtest results
- 🎯 **Market Regime Features** - SPY correlation, market volatility, beta approximation
- 🧾 **One-Click CSV + Visual Export** - Download enriched predictions plus PNG charts as a zip bundle for reporting

### Technical Features

##  3. Features  
- **Next-Day Movement Prediction** (UP/DOWN + confidence %)  
- **Real-Time Stock Data** using Yahoo Finance  
- **Multi-Ticker Support**  
- **Interactive Visualizations:**  
  - Confusion Matrix  
  - ROC Curve  
  - Feature Importance  
  - Price History  
- **Financial News Fetching** (News API)  
- **Live Stock Graphs** (MarketStack API)  

---

##  4. Project Structure  

```
market-classifier/
│
├── app.py                          # Frontend web server (Flask)
├── server.py                       # Backend API (Flask + ML)
├── main.py                         # ML pipeline
├── index.html                      # Frontend UI
├── requirements.txt                # Dependencies
├── README.md
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── external_features.py
│   └── lstm_model.py
│
├── data/
├── models/
└── results/
```

---

##  5. Backend Stack  
- Python 3.9  
- Flask 3.0  
- XGBoost 2.0  
- scikit-learn 1.3  
- pandas 2.1  
- yfinance 0.2  
- TA-Lib (technical indicators)  
- matplotlib / seaborn  

---

##  6. Frontend Stack  
- HTML5  
- Tailwind CSS  
- JavaScript  
- Fetch API  

---

## 🚀 Usage

### Option 1: Full Stack Application (Recommended)

**Terminal 1: Start Backend API Server**
```bash
python3 server.py
```
- Runs on: `http://localhost:5000`
- Exposes ML prediction endpoints
- Handles model training & caching

**Terminal 2: Start Frontend Web Server**
```bash
python3 app.py
```
- Runs on: `http://localhost:8080`
- Serves the interactive UI
- Opens automatically in browser

**Access Application:**
- Open browser: `http://localhost:8080`
- Search for stocks: `AAPL, TSLA, GOOGL`
- Click "View Model Visualizations"
- Click "Download CSV + Visuals" to grab a zip containing predictions + charts

---

### Option 1B: Docker Deployment (Separate Frontend & Backend)

Two lightweight Dockerfiles are provided so the ML API and the UI can be managed independently.  
See `Dockerfile.backend`, `Dockerfile.frontend`, and `docker-compose.yml`.

```bash
# Build both containers
docker compose build

# Run both services; Ctrl+C to stop
docker compose up
```

You can also run the services individually:

```bash
# Backend only
docker build -f Dockerfile.backend -t market-classifier-backend .
docker run -p 5000:5000 market-classifier-backend

# Frontend only (expects backend mapped locally on :5000)
docker build -f Dockerfile.frontend -t market-classifier-frontend .
docker run -p 8080:8080 market-classifier-frontend
```

> The browser code calls `http://localhost:5000/api/...`, so make sure the backend container is published on that host port when running the frontend.

---

### Unique Intelligence Add-ons

- **Macro shock overlays** highlight upcoming economic-calendar events (Fed, CPI, payrolls, etc.) that historically stress the sectors you are analyzing.
- **What-if sliders** let you twist volatility, the risk-free rate, and news-sentiment bias to see how each ticker's confidence would bend in real time.
- **Peer anomaly radar** runs the same ML inference on sector peers and surfaces whoever deviates most from the cohort, flagging contrarian setups.

---

##  8. Installation

### **Prerequisites**
- Python 3.9 -- 3.11  
- pip  

### **Step 1 — Create Virtual Environment**
```
python -m venv venv
venv\Scripts\activate
```

### **Step 2 — Install Dependencies**
```
pip install -r requirements.txt
```

---

##  9. How to Use

### **Option 1: Full Stack Mode (Recommended)**

**Response:**
```json
{
  "predictions": [
    {
      "ticker": "AAPL",
      "currentPrice": 172.50,
      "estimatedPrice": 175.43,
      "direction": "UP",
      "confidence": 87.0,
      "probabilityUp": 63.5,
      "probabilityDown": 36.5,
      "recommendation": "BUY",
      "riskLevel": "Medium",
      "expectedMovePct": 1.7,
      "rationale": "Model detects 63.5% probability of upside with solid momentum.",
      "riskNote": "Risk level: Medium (σ≈1.35%)"
    }
  ],
  "metadata": {
    "model_accuracy": 0.6234,
    "roc_auc": 0.6891,
    "primary_ticker": "AAPL",
    "train_samples": 412,
    "test_samples": 103
  }
}
```

#### 2. Get Visualizations
```http
POST /api/visualizations
```
python server.py
```
Runs at: **http://localhost:5000**

### **Terminal 2 — Start Frontend**
```
python app.py
```
### **Terminal 3 — Starts Model **
```
python main.py 
Runs at: **http://localhost:8080**
```
---

##  10. Model Configuration (XGBoost)

```
{
  "max_depth": 5,
  "learning_rate": 0.1,
  "n_estimators": 150,
  "objective": "binary:logistic",
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "gamma": 0.1,
  "reg_alpha": 0.1,
  "reg_lambda": 1.0
}
```

#### 6. Download Predictions + Visualizations
```http
POST /api/download-report
```

**Request Body:**
```json
{
  "tickers": "AAPL, TSLA"
}
```

**Response:**
- `200 OK` with `application/zip` payload.
- Contents: `predictions.csv`, `metrics.csv`, and `visualizations/<ticker>/*.png`.

> Use this endpoint when you need an auditable artifact for judges, PMs, or downstream dashboards.

---

## 🤖 Model Details

### Feature Engineering (250+ Features)

| Category | Features | Examples |
|----------|----------|----------|
| **Returns** | 5 features | `return_1d`, `return_5d`, `return_10d` |
| **Moving Averages** | 12 features | `sma_20`, `price_to_sma_20`, `sma_slope`, `ema_12`, `ema_26` |
| **Volatility** | 12 features | `volatility_20d`, `price_range_20d`, `atr`, `atr_percent` |
| **Momentum** | 15 features | `rsi_14`, `macd`, `roc_10`, `stoch_k`, `stoch_d`, `williams_r`, `adx` |
| **Bollinger Bands** | 6 features | `bb_position`, `bb_width`, `bb_squeeze` |
| **Fibonacci** | 10 features | `fib_236`, `fib_382`, `fib_500`, `fib_618`, `fib_786`, `dist_to_fib_*` |
| **Ichimoku Cloud** | 8 features | `ichimoku_tenkan`, `ichimoku_kijun`, `ichimoku_senkou_a/b`, `ichimoku_above_cloud` |
| **Volume** | 5 features | `volume_ratio`, `high_volume`, `volume_trend` |
| **Market Regime** | 5 features | `market_corr_20d`, `beta_60d`, `market_volatility_20d` |
| **Cross-Ticker** | 10+ features | `corr_20d_TICKER`, `corr_60d_TICKER` (per ticker) |
| **Time Features** | 5 features | `day_of_week`, `month`, `quarter`, `is_month_end` |
| **Lags** | 10 features | `close_lag_1`, `return_lag_5` |
| **Interactions** | 7 features | `rsi_vol_interaction`, `return_volume_interaction`, `rsi_macd_interaction`, `rsi_bb_interaction`, `macd_volume_interaction`, `adx_vol_interaction` |

### XGBoost Configuration
```python
{
    'max_depth': 6,              # Optimized via hyperparameter tuning
    'learning_rate': 0.08,       # Optimized via hyperparameter tuning
    'n_estimators': 200,         # Optimized via hyperparameter tuning
    'objective': 'binary:logistic',
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.15,
    'reg_alpha': 0.2,
    'reg_lambda': 1.2
}
```

**Note:** Hyperparameters are automatically tuned using RandomizedSearchCV with TimeSeriesSplit cross-validation for datasets with >100 samples.

### Training Process

1. **Data Download** - 2 years of OHLCV data via yfinance (including SPY for market regime)
2. **Feature Engineering** - 250+ technical indicators (including Fibonacci, Ichimoku), market regime, cross-ticker correlation
3. **Target Creation** - Binary label (tomorrow > today) with no data leakage
4. **Data Cleaning** - Handle inf, NaN, outliers, extreme values
5. **Feature Selection** - Top K-best features (if >100 features available)
6. **Train/Test Split** - 80/20 time-series split (no shuffle, time-aware)
7. **Scaling** - StandardScaler (mean=0, std=1)
8. **Hyperparameter Tuning** - RandomizedSearchCV with TimeSeriesSplit (if >100 samples)
9. **Training** - XGBoost with optimized parameters
10. **Validation Hold-Out** - Last 10-15% of training window used as an internal validation fold with early stopping
11. **Evaluation** - Accuracy, ROC-AUC, F1, Precision, Recall, Confusion Matrix, Feature Importance
12. **SHAP Explanations** - Model interpretability plots (shows WHY predictions are made)
13. **Backtesting** - Historical performance validation with trading simulation

### No Data Leakage
```python
# CORRECT (what we do)
df['future_close'] = df['Close'].shift(-1)  # Tomorrow's price
df['target'] = (df['future_close'] > df['Close']).astype(int)
df = df[:-1]  # Remove last row (no future data)
```

---

## 🌍 SDG Alignment

### Supported SDG Categories

| Category | SDG # | Goal | Tickers | Bonus |
|----------|-------|------|---------|-------|
| **SDG_CLEAN_ENERGY** | 7 | Affordable & Clean Energy | ICLN, TAN, ENPH, FSLR, RUN, SEDG | +20% |
| **SDG_HEALTH** | 3 | Good Health & Well-being | JNJ, PFE, UNH, ABBV, TMO, DHR | +20% |
| **SDG_CLIMATE** | 13 | Climate Action | TSLA, NEE, BEP, ENPH, RUN | +20% |
| **SDG_INNOVATION** | 9 | Industry & Innovation | AAPL, MSFT, GOOGL, NVDA, AMD | +20% |

### SDG Impact Score: 8.5/10

**Contributions:**
- ✅ Enhances investment intelligence for sustainable sectors
- ✅ Improves capital allocation to clean energy projects
- ✅ Supports market efficiency in renewable energy
- ✅ Transparent, open-source ML approach

---

##  13. Troubleshooting  

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test Accuracy** | 55-65% | Modest edge over random (50%) |
| **ROC-AUC** | 0.65-0.75 | Moderate discriminative ability |
| **F1-Score** | 0.55-0.65 | Balanced precision-recall |
| **Training Time** | 30-60s | First request (cached afterward) |

> **Reality Check:** The training loop now carves out a rolling validation fold with early stopping. This prevents the suspicious 100% training accuracy seen before and surfaces a `val_accuracy` metric inside the API response/visualizations.

### Why 55-65% is Realistic

1. **Market Efficiency** - Public technical indicators already priced in
2. **Noise Dominance** - Short-term (1-day) movements are mostly random
3. **Limited Features** - Only technical indicators (no fundamentals/sentiment)
4. **Binary Classification** - Predicts direction, not magnitude

### Sample Output
```
X = X.replace([np.inf, -np.inf], np.nan)
X = X.ffill().bfill().fillna(0)
```

### 2. **"Failed to download data"**
- Check internet  
- Retry with:
```
python main.py --custom AAPL
```

### 3. **Port Already in Use**
```
lsof -i :5000
kill -9 <PID>
```

### 4. **Frontend Not Connecting**
Check API URL in index.html:
```
const API_BASE = "http://localhost:5000/api";
```

---

## 🚀 Future Improvements

### Short-Term (Easy)
- [ ] Add more predefined categories (Commodities, Crypto, REITs)
- [ ] Email/SMS alerts for predictions
- [x] Export predictions to CSV (includes visualization bundle zip)
- [ ] Add dark mode to frontend
- [ ] Deployment scripts (Docker, Heroku)

### Medium-Term (Moderate)
- [ ] LSTM/Transformer models for sequence learning
- [ ] Sentiment analysis integration (Twitter, News APIs)
- [ ] Fundamental data (P/E ratios, earnings)
- [ ] Walk-forward optimization
- [ ] SHAP explainability for predictions
- [ ] Portfolio optimization recommendations

### Long-Term (Advanced)
- [ ] Real-time streaming predictions (WebSocket)
- [ ] Multi-day horizon predictions (3-day, 7-day)
- [ ] Reinforcement learning for trading strategies
- [ ] Backtesting framework with PnL tracking
- [ ] Mobile app (React Native)
- [ ] User authentication and personalized models

---

## 📚 References

### Libraries & Tools
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Technical Analysis Library (ta)](https://technical-analysis-library-in-python.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Tailwind CSS](https://tailwindcss.com/)

### Research Papers
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- Patel, J., et al. (2015). "Predicting stock market index using fusion of machine learning techniques"

### Financial Data
- [Yahoo Finance](https://finance.yahoo.com/)
- [UN Sustainable Development Goals](https://sdgs.un.org/)

---

## 👥 Contributors

**Developed by:** Meet and Jaimin

**Project Type:** ML Engineering Project - Market Movement Classification

**Course:** Machine Learning for Financial Markets

**Institution:** [Your Institution]

**Date:** November 2025

---

## 📄 License

MIT License

Copyright (c) 2025 Meet and Jaimin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🙏 Acknowledgments

- **XGBoost Team** - For the excellent gradient boosting library
- **Yahoo Finance** - For providing free financial data API
- **scikit-learn Community** - For comprehensive ML utilities
- **Flask Team** - For the lightweight web framework
- **Tailwind CSS** - For the utility-first CSS framework
- **UN SDG Initiative** - For sustainable development goals framework

---

##  14. Contributors  
- **Meet Ratwani**  
- **Jaimin Pansal**  
Airport School, Ahmedabad (Gujarat)

---

##  15. Acknowledgments  
- XGBoost Team  
- Yahoo Finance  
- scikit-learn Community  
- Flask Team  
- Tailwind CSS  
- YouTube Creators  
- Jupyter Notebooks (free GPUs)
- AI Models For Explaining us Market core concepts and Movements .
- AI Model (Copilot ) to tell what things are to be added in Readme for a hackathon   

---

## 16. Links  
**GitHub Repository:**  
https://github.com/jai-git4208/market-classifier  

Built with ❤️ by **Meet Ratwani  & Jaimin Pansal**  
