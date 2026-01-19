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

# --- DATA LOADING FUNCTIONS ---
def load_data():
    feature_path = os.path.join('features', 'btc_usdt_1h_features.parquet')
    if os.path.exists(feature_path):
        return pd.read_parquet(feature_path)
    return pd.DataFrame()

def load_model():
    model_path = os.path.join('models', 'training', 'btc_usdt_1h_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_trade_history():
    log_path = os.path.join('logs', 'trade_history.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return []

def load_system_logs():
    log_path = os.path.join('logs', 'system_memory.csv')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        # Ensure new columns exist if reading old CSV
        if 'anomaly_score' not in df.columns:
            df['anomaly_score'] = 0.0
        if 'network_gas_gwei' not in df.columns:
            df['network_gas_gwei'] = 20.0 # Default fallback
        return df
    return pd.DataFrame()

# --- MAIN DASHBOARD LAYOUT ---
st.title("BTC/USDT AI-Driven Trading Pipeline 🚀")
st.markdown("### Hybrid AI: LightGBM + LSTM Deep Learning + NLP + Web3 Analytics")

df = load_data()
model = load_model()

# Sidebar for manual refresh
if st.sidebar.button("Refresh Data"):
    st.rerun()

if df.empty or model is None:
    st.error("Data or Model not found. Please run the training scripts first.")
else:
    col1, col2 = st.columns(2)

    # --- TOP LEFT: Price Chart ---
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
        
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # --- TOP RIGHT: AI Decision Engine ---
    with col2:
        st.subheader("AI Decision Engine (Live)")
        
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
        m1.metric(label="LightGBM Prediction", 
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
        
        st.bar_chart(importance.set_index('feature'), height=250)

    # --- MIDDLE: Trade History ---
    st.subheader("Simulated Trade Log (Paper Trading)")
    history = load_trade_history()
    if history:
        # Convert to DF and sort by newest first
        log_df = pd.DataFrame(history)
        log_df = log_df.iloc[::-1] # Reverse order
        st.dataframe(log_df, use_container_width=True, height=200)
    else:
        st.info("No trades executed yet.")

    # --- BOTTOM: System Intelligence (MLOps + Deep Learning + Web3 + Sentiment History) ---
    sys_df = load_system_logs()

    if not sys_df.empty:
        # Convert timestamp to datetime
        sys_df['timestamp'] = pd.to_datetime(sys_df['timestamp'])
        
        st.markdown("---")
        st.subheader("📡 System Intelligence & Network Health")
        
        # --- ROW 1: The 3 Critical Gauges ---
        c1, c2, c3 = st.columns([1, 1.5, 1]) 
        
        # 1. MLOps Confidence
        with c1:
            st.write("**Model Confidence**")
            st.line_chart(sys_df.set_index('timestamp')['model_confidence'], height=250)
            
        # 2. Deep Learning Gauge
        with c2:
            last_anom = sys_df['anomaly_score'].iloc[-1]
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = last_anom,
                title = {'text': "LSTM Anomaly Score (Risk)"},
                gauge = {
                    'axis': {'range': [0, 0.1]},
                    'bar': {'color': "red" if last_anom > 0.05 else "green"},
                    'steps': [
                        {'range': [0, 0.05], 'color': "lightgreen"},
                        {'range': [0.05, 0.1], 'color': "salmon"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.05
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=0, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
            if last_anom > 0.05:
                st.error("🚨 CRITICAL: Anomaly Detected")
            else:
                st.success("✅ Deep Learning: Normal")

        # 3. Web3 Gas Monitor
        with c3:
            st.write("**On-Chain Activity (Gas)**")
            last_gas = sys_df['network_gas_gwei'].iloc[-1]
            st.metric(label="Ethereum Gas (Gwei)", value=f"{last_gas:.1f}", 
                     delta="Congested" if last_gas > 50 else "Normal", delta_color="normal")
            st.bar_chart(sys_df.set_index('timestamp')['network_gas_gwei'].tail(20), height=150)

        # --- ROW 2: The Sentiment History ---
        st.write("---")
        st.write("**📰 News Sentiment vs. Bitcoin Price**")
        
        fig_sent = go.Figure()
        # News Sentiment (Bar)
        fig_sent.add_trace(go.Bar(
            x=sys_df['timestamp'], y=sys_df['news_sentiment'],
            name='Sentiment', marker_color='orange', opacity=0.6
        ))
        # Price (Line) - Secondary Axis
        fig_sent.add_trace(go.Scatter(
            x=sys_df['timestamp'], y=sys_df['btc_price'],
            name='BTC Price', yaxis='y2', line=dict(color='blue')
        ))
        fig_sent.update_layout(
            height=350, 
            yaxis=dict(title="Sentiment (-1 to 1)"),
            yaxis2=dict(title="Price ($)", overlaying='y', side='right'),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0, y=1.1, orientation='h')
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        st.caption(f"Last System Update: {sys_df['timestamp'].iloc[-1]}")

    else:
        st.warning("No system logs found yet. Run main_loop.py to generate data.")
