# Market Movement Classifier  

The **Market Movement Classifier** is an AI-based stock prediction system developed by **Team Aimers**.  
It uses a trained **XGBoost model** to predict next-day stock movements (UP/DOWN) and provides **real-time charts**, **latest stock data**, and **financial news** through an integrated full-stack web application.

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

---

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

##  7. AI Model  
- **150+ engineered features**  
- Technical Indicators (RSI, MACD, EMA, etc.)  
- Regime-based features (SPY, VIX)  
- Data cleaning & outlier handling  
- Model caching for faster inference  

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

### **Terminal 1 — Start Backend**
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

---

##  11. Training Process  
1. Download 2-year OHLCV data  
2. Build 150+ technical indicators  
3. Create binary target label  
4. Clean: NaN, inf, outliers  
5. 80/20 time-series train-test split  
6. Scale with StandardScaler  
7. Train XGBoost with cross-validation  
8. Evaluate using Accuracy, ROC-AUC, CM  

---

##  12. SDG Alignment (Bonus)  
The model can also analyse **Clean Energy Stocks (SDG #7)** such as `ADANIGREEN.NS`.  
Dedicated clean-energy datasets & models are included.

---

##  13. Troubleshooting  

### 1. **"Input contains infinity"**
Fixed using:
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

### 5. **Training Too Slow**
Reduce dataset:
```
period='1y'
```
Or limit features:
```
SelectKBest(k=30)
```

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
