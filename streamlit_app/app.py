import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import json
import sys

# Allow importing from modules folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.sentiment_analysis import SentimentPlugin

st.set_page_config(page_title="BTC AI Trader", layout="wide")

def load_data():
    feature_path = os.path.join('features', 'btc_usdt_1h_features.parquet')
    df = pd.read_parquet(feature_path)
    return df

def load_model():
    model_path = os.path.join('models', 'training', 'btc_usdt_1h_model.joblib')
    model = joblib.load(model_path)
    return model

def load_trade_history():
    log_path = os.path.join('logs', 'trade_history.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return []

st.title("BTC/USDT AI-Driven Trading Pipeline 🚀")
st.markdown("### DePIN-Integrated De-Fi Trading System with NLP")

df = load_data()
model = load_model()

# Sidebar for manual refresh
if st.sidebar.button("Refresh Data"):
    st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Action & Indicators")
    last_168 = df.tail(168) 
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=last_168['timestamp'],
                open=last_168['open'], high=last_168['high'],
                low=last_168['low'], close=last_168['close'],
                name='BTC/USDT'))
    
    fig.add_trace(go.Scatter(x=last_168['timestamp'], y=last_168['upper_band'], name='Upper BB', line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=last_168['timestamp'], y=last_168['lower_band'], name='Lower BB', line=dict(color='gray', width=1)))
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AI Decision Engine")
    
    # 1. ML Signal
    features = [c for c in df.columns if c not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
    latest_row = df.iloc[[-1]][features]
    prediction = model.predict(latest_row)[0]
    prob = model.predict_proba(latest_row)[0][1]
    
    # 2. Sentiment Signal
    sent = SentimentPlugin()
    news_sig, news_score = sent.get_sentiment_signal()
    
    # Display Metrics side-by-side
    m1, m2 = st.columns(2)
    m1.metric(label="ML Model Prediction", 
              value="BUY (UP)" if prediction == 1 else "SELL/HOLD", 
              delta=f"{prob:.2%} Conf")
    
    m2.metric(label="News Sentiment (NLP)", 
              value=f"{news_score:.2f}", 
              delta="BULLISH" if news_sig == 1 else "BEARISH" if news_sig == -1 else "NEUTRAL",
              delta_color="normal")
    
    st.write("---")
    st.write("**Feature Importance (Explainable AI)**")
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).head(10)
    
    st.bar_chart(importance.set_index('feature'))

st.subheader("Simulated Trade Log (Paper Trading)")
history = load_trade_history()
if history:
    # Convert to DF and sort by newest first
    log_df = pd.DataFrame(history)
    log_df = log_df.iloc[::-1] # Reverse order
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("No trades executed yet.")
