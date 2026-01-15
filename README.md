# BTC/USDT AI-Driven Trading Pipeline 🚀

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/AI-LightGBM%20%2B%20SHAP-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview
A modular, explainable AI trading system for Bitcoin (BTC/USDT) that bridges **Quantitative Finance** and **DePIN (Decentralized Physical Infrastructure)** concepts. 

Unlike black-box trading bots, this pipeline emphasizes **Explainable AI (XAI)**. It uses **LightGBM** for predictive modeling and **SHAP (Shapley Additive Explanations)** to visualize exactly *why* the model enters a trade—whether it's driven by technical momentum or simulated on-chain network activity.

## ✨ Key Features (Resume Highlights)

### 🧠 Explainable ML Engine
- **Model:** Gradient Boosting (LightGBM) trained on historical Binance data.
- **Transparency:** Integrated `shap` library to generate feature importance plots for every prediction, solving the "black box" problem in algorithmic trading.

### 🔗 DePIN & On-Chain Integration
- **Simulation:** Includes a custom plugin that simulates **DePIN** metrics (Active Addresses, Network Hashrate) to augment price data with fundamental blockchain health signals.
- **Hypothesis:** Blending price action (Technical Analysis) with Network Usage (Fundamental Analysis) yields higher Sharpe ratios.

### 📊 Full-Stack Dashboard
- **Tech Stack:** Streamlit + Plotly.
- **Capabilities:** Real-time visualization of equity curves, model confidence levels, and live buy/sell signals.

### 🛠 Engineering Best Practices
- **Modular Architecture:** Plugins (e.g., Sentiment Analysis, Risk Manager) are decoupled from the core execution engine.
- **Idempotency:** Trade execution uses UUIDs to prevent duplicate orders during network retries.
- **Safe Paper Trading:** Fully simulated execution environment with JSON logging.

## Quick Start

### 1. Installation
```bash
# Clone the repo
git clone [https://github.com/MohitSingh-2335/btc-ai-trading-pipeline.git](https://github.com/MohitSingh-2335/btc-ai-trading-pipeline.git)
cd btc-ai-trading-pipeline

# Install dependencies (Linux/WSL2 recommended for TA-Lib)
pip install -r requirements.txt

### 2. Run the Pipeline

# Step 1: Fetch and Clean Data
python3 scripts/ingest_clean.py

# Step 2: Generate Features (TA-Lib + DePIN metrics)
python3 scripts/feature_engine.py

# Step 3: Train Model
python3 scripts/train_backtest.py

# Step 4: Launch Dashboard
streamlit run streamlit_app/app.py

### Project Structure

btc_trading_pipeline/
├── data/               # Parquet files (Raw & Cleaned)
├── features/           # Feature engineered datasets
├── models/             # Trained LightGBM models & SHAP explainers
├── modules/            # Plugins (Sentiment, On-Chain)
├── scripts/            # Core logic (Ingest, Train, Trade)
├── streamlit_app/      # Visualization Dashboard
└── logs/               # Trade history and system logs

### Future Roadmap:

[ ] Connect to live Web3.py nodes for real-time DePIN data.

[ ] Implement LSTM Autoencoders for anomaly detection.

[ ] Containerize with Docker for cloud deployment.

---

