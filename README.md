# 🤖 AI-Powered Crypto Trading System (BTC/USDT)

A production-grade algorithmic trading bot that combines **Gradient Boosting (LightGBM)** for directional prediction, **Deep Learning (LSTM Autoencoder)** for anomaly detection, and **Real-Time Sentiment Analysis (NLP)**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/ML-LightGBM-green)
![PyTorch](https://img.shields.io/badge/DL-PyTorch-orange)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

## 🧠 System Architecture

The system operates on a 60-second autonomous cycle:

1.  **Data Ingestion:** Fetches OHLCV data from Binance (Futures).
2.  **Feature Engineering:** Calculates 50+ technical indicators (RSI, MACD, Bollinger Bands).
3.  **Hybrid Intelligence:**
    * **LightGBM:** Predicts short-term price direction (Up/Down).
    * **LSTM Autoencoder:** Detects market anomalies/crashes (Risk Management).
    * **VADER NLP:** Scrapes CoinTelegraph RSS for real-time news sentiment.
    * **Web3 Monitor:** Checks Ethereum Gas fees to gauge network congestion.
4.  **Execution:** Paper trades based on a weighted consensus of all models.
5.  **MLOps:** Automatically retrains the LightGBM model when 50+ new data points are collected.

## 📂 Project Structure

├── data/               # Raw and processed parquet files
├── features/           # Engineered features for ML
├── logs/               # Trade history (JSON) and System memory (CSV)
├── models/             # Trained LightGBM and PyTorch models
├── modules/            # AI & Data Plugins
│   ├── anomaly_detector.py  # LSTM Autoencoder
│   ├── sentiment_analysis.py # RSS + VADER
│   └── on_chain_data.py     # Web3 Gas Monitor
├── scripts/            # Pipeline Scripts
│   ├── ingest_clean.py      # Data fetching (Pagination)
│   ├── feature_engine.py    # Indicator calculation
│   ├── train_model.py       # Model training
│   ├── risk_trade.py        # Main Trading Logic
│   └── auto_retrain.py      # MLOps Pipeline
├── streamlit_app/      # Real-time Dashboard
├── config.yaml         # Central Configuration
└── main_loop.py        # System Orchestrator

## 🚀 How to Run

### Install Dependencies:
pip install -r requirements.txt

### Initialize Data:
python3 scripts/ingest_clean.py
python3 scripts/feature_engine.py
python3 scripts/train_model.py

### Start the Bot:
python3 main_loop.py

### Launch Dashboard:
streamlit run streamlit_app/app.py

## ⚙️ Configuration

Edit config.yaml to adjust trading parameters:

trading:
  symbol: "BTC/USDT"
  min_confidence: 0.55
risk:
  sentiment_threshold: -0.2
  max_gas_gwei: 50

## 📊 Dashboard

The system includes a Streamlit dashboard for real-time monitoring of:
Live Price & ML Confidence
Sentiment Score Analysis
Anomaly Detection Score
Simulated Trade History
