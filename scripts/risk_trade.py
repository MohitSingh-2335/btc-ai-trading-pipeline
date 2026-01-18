import pandas as pd
import joblib
import os
import json
import sys
import uuid
from datetime import datetime

# Add path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.sentiment_analysis import SentimentPlugin
from modules.anomaly_detector import MarketAnomalyDetector

def load_artifacts(filename):
    model_path = os.path.join('models', 'training', f"{filename}_model.joblib")
    features_path = os.path.join('features', f"{filename}_features.parquet")
    
    model = joblib.load(model_path)
    df = pd.read_parquet(features_path)
    return model, df

def get_latest_signal(model, df):
    # Exclude non-feature columns
    features = [c for c in df.columns if c not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
    latest_row = df.iloc[[-1]][features]
    prediction = model.predict(latest_row)[0]
    probability = model.predict_proba(latest_row)[0][1]
    return prediction, probability, df.iloc[-1]['close']

def log_system_state(timestamp, price, sentiment, prediction, probability, signal_type, anomaly_score):
    """
    Saves a snapshot, including the new Deep Learning Anomaly Score.
    """
    log_path = os.path.join('logs', 'system_memory.csv')
    
    new_row = {
        'timestamp': timestamp,
        'btc_price': price,
        'news_sentiment': sentiment,
        'model_prediction': prediction,
        'model_confidence': probability,
        'signal_type': signal_type,
        'anomaly_score': anomaly_score  # <--- NEW FIELD
    }
    
    # Check if header is needed (if file doesn't exist)
    header = not os.path.exists(log_path)
    
    # Append to CSV
    pd.DataFrame([new_row]).to_csv(log_path, mode='a', header=header, index=False)
    print(f"📝 Data Recorded (Anomaly Score: {anomaly_score:.4f})")

def execute_paper_trade(signal, price, size, sentiment_score, anomaly_score):
    trade_id = str(uuid.uuid4())
    trade_data = {
        "id": trade_id,
        "timestamp": datetime.now().isoformat(),
        "action": "BUY" if signal == 1 else "SELL",
        "price": price,
        "size_usdt": size,
        "sentiment_score": sentiment_score,
        "anomaly_score": anomaly_score,
        "status": "FILLED"
    }
    
    log_path = os.path.join('logs', 'trade_history.json')
    
    history = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                history = json.load(f)
            except:
                pass
    
    history.append(trade_data)
    
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"✅ Trade EXECUTED: BUY ${size:.2f} BTC at ${price:.2f}")

if __name__ == "__main__":
    print("--- Starting AI Trading Cycle ---")
    
    # 1. Load Data & ML Model
    model, df = load_artifacts("btc_usdt_1h")
    ml_pred, ml_prob, current_price = get_latest_signal(model, df)
    
    # 2. Get News Sentiment (NLP)
    sentiment_plugin = SentimentPlugin()
    news_signal, news_score = sentiment_plugin.get_sentiment_signal()
    
    # 3. Get Market Anomaly (Deep Learning) <-- NEW STEP
    detector = MarketAnomalyDetector()
    # Train on last 100 candles to learn "normal"
    recent_prices = df['close'].values[-100:]
    detector.train_on_history(recent_prices)
    # Check last 30 candles for anomalies
    anom_score, is_anomaly = detector.detect_anomaly(df['close'].values[-30:])
    
    current_time = datetime.now().isoformat()
    final_action = "HOLD"

    print(f"Price: ${current_price:.2f} | ML Conf: {ml_prob:.2f}")
    print(f"Sentiment: {news_score:.2f} | Anomaly Score: {anom_score:.4f}")

    # 4. Multi-Stage Decision Logic
    if is_anomaly:
        print("🚨 CIRCUIT BREAKER TRIGGERED: High Market Anomaly Detected. Trading Halted.")
        final_action = "HALTED_ANOMALY"
        
    elif ml_pred == 1:
        if news_signal == -1:
            print("Risk Manager: Trade BLOCKED (Negative News).")
            final_action = "BLOCKED_NEWS"
        else:
            size = 10000 * 0.02 
            execute_paper_trade(1, current_price, size, news_score, anom_score)
            final_action = "BUY_EXECUTED"
    else:
        print("Model says HOLD/SELL.")
        final_action = "HOLD"

    # 5. Record Memory
    log_system_state(current_time, current_price, news_score, ml_pred, ml_prob, final_action, anom_score)
