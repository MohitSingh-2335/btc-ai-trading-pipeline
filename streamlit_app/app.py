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

def load_system_logs():
    log_path = os.path.join('logs', 'system_memory.csv')
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame()

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

st.markdown("---")
st.subheader("🛠 MLOps: Model Health & Drift Monitoring")

sys_df = load_system_logs()

if not sys_df.empty:
    # Convert timestamp to datetime
    sys_df['timestamp'] = pd.to_datetime(sys_df['timestamp'])
    
    # Create 2 columns for MLOps charts
    m1, m2 = st.columns(2)
    
    with m1:
        st.write("**Model Confidence Over Time**")
        # Plot Confidence
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=sys_df['timestamp'], 
            y=sys_df['model_confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#00ff00', width=2)
        ))
        fig_conf.update_layout(height=300, yaxis_title="Probability (0-1)", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_conf, use_container_width=True)
        
    with m2:
        st.write("**News Sentiment vs. Price**")
        # Plot Sentiment vs Price
        fig_sent = go.Figure()
        
        # News Sentiment (Bar)
        fig_sent.add_trace(go.Bar(
            x=sys_df['timestamp'], 
            y=sys_df['news_sentiment'],
            name='Sentiment',
            marker_color='orange',
            opacity=0.6
        ))
        
        # Price (Line) - Secondary Axis
        fig_sent.add_trace(go.Scatter(
            x=sys_df['timestamp'], 
            y=sys_df['btc_price'],
            name='BTC Price',
            yaxis='y2',
            line=dict(color='blue')
        ))
        
        fig_sent.update_layout(
            height=300, 
            yaxis=dict(title="Sentiment (-1 to 1)"),
            yaxis2=dict(title="Price ($)", overlaying='y', side='right'),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0, y=1.2, orientation='h')
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # Drift Metric
    avg_conf = sys_df['model_confidence'].tail(5).mean()
    st.info(f"📊 Recent Average Confidence (Last 5 Runs): **{avg_conf:.2%}**")
    if avg_conf < 0.52:
        st.error("⚠️ Warning: Model Confidence is Low. Market regime may be drifting.")
    else:
        st.success("✅ Model is operating with healthy confidence levels.")

else:
    st.warning("No system logs found yet. Run the main_loop.py to generate MLOps data.")
