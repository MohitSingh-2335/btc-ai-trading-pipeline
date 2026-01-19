# 📉 Quantitative AI Trading System (LSTM + LightGBM)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/AI-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Production-green)

## 📋 Overview
An institutional-grade algorithmic trading pipeline that combines **Gradient Boosting (LightGBM)** for directional prediction with **Deep Learning (LSTM Autoencoders)** for anomaly detection.

Unlike standard bots, this system features a **"Self-Healing" Architecture**: it continuously monitors its own confidence and market entropy, automatically halting trading during "Black Swan" events using an unsupervised neural network.

## ✨ Key Features

### 🧠 Dual-Model Architecture
1.  **Alpha Engine (LightGBM):** Analyzes technical indicators (RSI, Bollinger Bands, MACD) to predict short-term price direction.
2.  **Risk Engine (PyTorch LSTM):** An unsupervised Autoencoder that learns "normal" market structure. If reconstruction error spikes (High MSE), it triggers a circuit breaker to stop trading.

### 🛠 MLOps & Monitoring
- **Drift Detection:** Real-time tracking of model confidence scores to detect regime changes.
- **Explainable AI (SHAP):** breakdown of why every trade was taken (e.g., "Bought because Volume > 200% avg").
- **System Memory:** Logs every inference tick to `system_memory.csv` for future retraining (Continuous Learning).

### ⚡ Engineering Stack
- **Data:** CCXT (Binance API) + Pandas
- **Training:** Scikit-Learn + LightGBM
- **Deep Learning:** PyTorch
- **Dashboard:** Streamlit + Plotly (Real-time visualization)

## 🚀 Quick Start
```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the Autonomous Loop (Backend)
python3 main_loop.py

# 3. Launch Dashboard (Frontend)
streamlit run streamlit_app/app.py

