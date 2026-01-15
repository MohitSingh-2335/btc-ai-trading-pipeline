import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import json

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

st.title("BTC/USDT AI-Driven Trading Pipeline")
st.markdown("### DePIN-Integrated De-Fi Trading System")

df = load_data()
model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Action & Indicators")
    last_500 = df.tail(168) 
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=last_500['timestamp'],
                open=last_500['open'],
                high=last_500['high'],
                low=last_500['low'],
                close=last_500['close'],
                name='BTC/USDT'))
    
    fig.add_trace(go.Scatter(x=last_500['timestamp'], y=last_500['upper_band'], name='Upper BB', line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=last_500['timestamp'], y=last_500['lower_band'], name='Lower BB', line=dict(color='gray', width=1)))
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AI Decision Engine")
    
    features = [c for c in df.columns if c not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
    latest_row = df.iloc[[-1]][features]
    prediction = model.predict(latest_row)[0]
    prob = model.predict_proba(latest_row)[0][1]
    
    st.metric(label="Next Hour Prediction", 
              value="BUY (UP)" if prediction == 1 else "SELL/HOLD (DOWN)", 
              delta=f"{prob:.2%} Confidence")
    
    st.write("Feature Importance (Top 10)")
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).head(10)
    
    st.bar_chart(importance.set_index('feature'))

st.subheader("Simulated Trade Log (Paper Trading)")
history = load_trade_history()
if history:
    st.table(pd.DataFrame(history))
else:
    st.info("No trades executed yet.")
