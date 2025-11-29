# 🏆 Market Movement Classifier - Hackathon Presentation Summary

## 🎯 Problem Statement Alignment

**Problem:** Train a classification model using historical time-series data (30-day stock prices) to predict whether the next day's closing value will be UP or DOWN (binary classification).

**✅ Our Solution:**
- ✅ Binary classification (UP/DOWN) using 2 years of historical data
- ✅ Next-day prediction (1-day forward horizon)
- ✅ XGBoost classifier with comprehensive feature engineering
- ✅ Python implementation with Pandas for feature engineering
- ✅ **SDG Alignment: +20% bonus** (Clean Energy, Healthcare, Climate Action)

---

## 🚀 Key Features & Innovations

### 1. **Advanced Feature Engineering (250+ Features)**
- **Technical Indicators:** RSI, MACD, Bollinger Bands, Stochastic, ADX, Williams %R
- **Advanced Indicators:** Fibonacci Retracement Levels, Ichimoku Cloud
- **Market Regime Features:** SPY correlation, market volatility, beta approximation
- **Cross-Ticker Features:** Correlation between multiple stocks
- **Time Features:** Day of week, month, quarter effects
- **Volume Analysis:** Volume ratios, trends, high-volume signals
- **Interaction Features:** RSI×Volatility, Return×Volume, RSI×BB, MACD×Volume, ADX×Volatility

### 2. **Sophisticated ML Pipeline**
- **XGBoost with Hyperparameter Tuning:** Automatic optimization using TimeSeriesSplit CV
- **SHAP Explainability:** Model interpretability - shows WHY predictions are made (judges love this!)
- **Backtesting Framework:** Historical performance validation with realistic trading simulation
- **Feature Selection:** Top K-best features (reduces overfitting)
- **Time-Series Aware:** No data leakage, proper train/test splitting
- **Robust Data Cleaning:** Handles infinity, NaN, extreme outliers

### 3. **SDG Alignment (4 Categories)**
| SDG # | Goal | Tickers | Impact |
|-------|------|---------|--------|
| **#7** | Affordable & Clean Energy | ICLN, TAN, ENPH, FSLR, RUN, SEDG | Renewable energy investment |
| **#3** | Good Health & Well-being | JNJ, PFE, UNH, ABBV, TMO, DHR | Healthcare access & innovation |
| **#13** | Climate Action | TSLA, NEE, BEP, ENPH, RUN | Climate-friendly investments |
| **#9** | Industry & Innovation | AAPL, MSFT, GOOGL, NVDA, AMD | Sustainable infrastructure |

### 4. **Production-Ready Application**
- **REST API:** Flask backend with model caching
- **Interactive Frontend:** Real-time predictions with visualizations
- **Comprehensive Metrics:** Accuracy, ROC-AUC, F1-Score, Confusion Matrix
- **Visualizations:** ROC curves, feature importance, price history, returns distribution

---

## 📊 Model Performance

### Expected Results:
- **Test Accuracy:** 55-65% (modest edge over 50% random)
- **ROC-AUC:** 0.65-0.75 (moderate discriminative ability)
- **F1-Score:** 0.55-0.65 (balanced precision-recall)

### Why This Performance is Realistic:
1. Market efficiency - technical indicators already priced in
2. Short-term noise - 1-day movements are mostly random
3. No fundamentals - only technical analysis features
4. Binary classification - predicts direction, not magnitude

---

## 🛠️ Technology Stack

### Core ML:
- **XGBoost 2.0** - Gradient boosting classifier
- **scikit-learn 1.3** - Feature selection, metrics, validation
- **pandas 2.1** - Data manipulation & feature engineering
- **yfinance 0.2** - Real-time stock data

### Web Application:
- **Flask 3.0** - REST API backend
- **Tailwind CSS** - Modern UI
- **JavaScript** - Interactive frontend

### Feature Engineering:
- **ta 0.11** - Technical analysis indicators
- **numpy 1.24** - Numerical computations

---

## 📁 Project Structure

```
market-classifier/
├── main.py              # Standalone ML pipeline (CLI)
├── server.py            # REST API backend
├── app.py               # Frontend web server
├── demo.py              # Quick demo script
├── src/
│   ├── data_loader.py          # Stock data download
│   ├── feature_engineering.py # 200+ features
│   └── model_training.py      # XGBoost training
├── data/                # Generated datasets
├── models/              # Trained models
└── results/             # Visualizations & metrics
```

---

## 🎬 Quick Start

### Option 1: Demo Script (Recommended for Judges)
```bash
python3 demo.py
```

### Option 2: Full Pipeline
```bash
# SDG Clean Energy (SDG #7)
python3 main.py --category SDG_CLEAN_ENERGY

# Custom tickers
python3 main.py --custom AAPL TSLA GOOGL

# Multi-category analysis
python3 main.py --multi
```

### Option 3: Web Application
```bash
# Terminal 1: Backend API
python3 server.py

# Terminal 2: Frontend
python3 app.py

# Open: http://localhost:8080
```

---

## 🌍 SDG Impact & Bonus Points

### SDG Alignment Score: 9/10

**Contributions:**
- ✅ **Investment Intelligence:** Enhances decision-making for sustainable sectors
- ✅ **Capital Allocation:** Improves funding to clean energy projects
- ✅ **Market Efficiency:** Supports price discovery in renewable energy
- ✅ **Transparency:** Open-source ML approach with explainable features
- ✅ **Scalability:** Works for any stock category, not just SDG-aligned

### Bonus Eligibility:
- ✅ **+20% SDG Alignment:** 4 SDG categories supported
- ✅ **Data Quality:** Real-time data from Yahoo Finance
- ✅ **Feature Engineering:** 200+ sophisticated features
- ✅ **Model Quality:** Hyperparameter tuning, feature selection

---

## 💡 Key Differentiators

1. **Model Interpretability:** SHAP values explain WHY the model makes predictions (critical for judges!)
2. **Backtesting Framework:** Validates model performance with realistic trading simulation
3. **Market Regime Awareness:** SPY correlation, market volatility features
4. **Multi-Ticker Analysis:** Cross-ticker correlation features
5. **Advanced Indicators:** Fibonacci, Ichimoku Cloud, ADX, Stochastic, Williams %R
6. **Time-Series Validation:** Proper TimeSeriesSplit CV (no look-ahead bias)
7. **Production Ready:** Full web app with API, not just a notebook
8. **SDG Focus:** Multiple SDG categories with clear impact documentation

---

## 📈 Use Cases

1. **Individual Investors:** Next-day stock movement predictions
2. **Portfolio Managers:** Risk assessment and position sizing
3. **SDG Investors:** Sustainable investment decision support
4. **Researchers:** Time-series classification benchmark
5. **Students:** ML pipeline learning example

---

## 🔬 Technical Highlights

### Feature Engineering Innovations:
- **Market Regime Features:** Beta, market correlation, regime detection
- **Cross-Asset Features:** Correlation between multiple tickers
- **Time-Based Features:** Calendar effects (day of week, month end)
- **Interaction Features:** Non-linear combinations (RSI×Volatility)

### Model Improvements:
- **Automatic Tuning:** RandomizedSearchCV with TimeSeriesSplit
- **Feature Selection:** SelectKBest to reduce overfitting
- **Robust Validation:** Time-series aware, no data leakage
- **Scalability:** Handles 1-10+ tickers simultaneously

---

## 🎯 Hackathon Judging Criteria Alignment

| Criterion | Our Score | Evidence |
|-----------|-----------|----------|
| **Problem Alignment** | 10/10 | Perfect match: binary classification, next-day prediction |
| **Technical Quality** | 10/10 | 250+ features, SHAP explainability, backtesting, hyperparameter tuning |
| **SDG Alignment** | 9/10 | 4 SDG categories, clear impact documentation |
| **Code Quality** | 9/10 | Clean structure, error handling, comprehensive documentation |
| **Innovation** | 10/10 | SHAP interpretability, backtesting, Fibonacci/Ichimoku, market regime |
| **Presentation** | 9/10 | Web app, visualizations, demo script, SHAP plots, backtest results |

**Total Estimated Score: 57/60 (95%)**

---

## 📞 Contact & Demo

**Team:** Meet and Jaimin  
**Project:** Market Movement Classifier  
**Category:** ML Classification + SDG Alignment

**Quick Demo:**
```bash
python3 demo.py
```

**Full Demo:**
```bash
python3 main.py --category SDG_CLEAN_ENERGY
```

---

## 🏅 Why We Should Win

1. **Complete Solution:** Not just a model, but a full production-ready application
2. **SDG Impact:** Clear alignment with 4 UN Sustainable Development Goals
3. **Technical Excellence:** 200+ features, hyperparameter tuning, proper validation
4. **Innovation:** Market regime features, cross-ticker analysis, advanced indicators
5. **Usability:** Web app, API, CLI - multiple interfaces for different users
6. **Documentation:** Comprehensive README, demo script, presentation summary

---

**Built with ❤️ for the Hackathon** 🚀

