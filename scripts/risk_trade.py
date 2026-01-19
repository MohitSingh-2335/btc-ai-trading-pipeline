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
from modules.on_chain_data import OnChainPlugin  # <--- NEW IMPORT

def load_artifacts(filename):
    model_path = os.path.join('models', 'training', f"{filename}_model.joblib")
    features_path = os.path.join('features', f"{filename}_features.parquet")
    
    model = joblib.load(model_path)
    df = pd.read_parquet(features_path)
    return model, df

def get_latest_signal(model, df):
    features = [c for c in df.columns if c not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
    latest_row = df.iloc[[-1]][features]
    prediction = model.predict(latest_row)[0]
    probability = model.predict_proba(latest_row)[0][1]
    return prediction, probability, df.iloc[-1]['close']

def log_system_state(timestamp, price, sentiment, prediction, probability, signal_type, anomaly_score, gas_gwei):
    """
    Saves snapshot including Web3 Data
    """
    log_path = os.path.join('logs', 'system_memory.csv')
    
    new_row = {
        'timestamp': timestamp,
        'btc_price': price,
        'news_sentiment': sentiment,
        'model_prediction': prediction,
        'model_confidence': probability,
        'signal_type': signal_type,
        'anomaly_score': anomaly_score,
        'network_gas_gwei': gas_gwei # <--- NEW FIELD
    }
    
    # Check header
    header = not os.path.exists(log_path)
    pd.DataFrame([new_row]).to_csv(log_path, mode='a', header=header, index=False)
    print(f"📝 Memory Updated (Gas: {gas_gwei:.1f} Gwei)")

def execute_paper_trade(signal, price, size, sentiment_score, anomaly_score, gas_gwei):
    trade_id = str(uuid.uuid4())
    trade_data = {
        "id": trade_id,
        "timestamp": datetime.now().isoformat(),
        "action": "BUY" if signal == 1 else "SELL",
        "price": price,
        "size_usdt": size,
        "sentiment_score": sentiment_score,
        "anomaly_score": anomaly_score,
        "network_gas": gas_gwei,
        "status": "FILLED"
    }
    
    log_path = os.path.join('logs', 'trade_history.json')
    history = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try: history = json.load(f)
            except: pass
    
    history.append(trade_data)
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"✅ Trade EXECUTED: BUY ${size:.2f} BTC at ${price:.2f}")

if __name__ == "__main__":
    print("--- Starting Multi-Modal AI Cycle ---")
    
    # 1. Technical (LightGBM)
    model, df = load_artifacts("btc_usdt_1h")
    ml_pred, ml_prob, current_price = get_latest_signal(model, df)
    
    # 2. Sentiment (NLP)
    try:
        sentiment_plugin = SentimentPlugin()
        # We capture both Signal (Direction) and Score (Intensity)
        news_signal, news_score = sentiment_plugin.get_sentiment_signal()
    except Exception as e:
        print(f"⚠️ NLP Error: {e} (Defaulting to Neutral)")
        news_signal, news_score = 0, 0.0
    
    # 3. Risk (LSTM)
    detector = MarketAnomalyDetector()
    recent_prices = df['close'].values[-100:]
    detector.train_on_history(recent_prices)
    anom_score, is_anomaly = detector.detect_anomaly(df['close'].values[-30:])
    
    # 4. On-Chain (Web3) <--- NEW STEP
    web3_plugin = OnChainPlugin()
    gas_gwei, net_status = web3_plugin.get_network_health()
    
    current_time = datetime.now().isoformat()
    final_action = "HOLD"

    print(f"Price: ${current_price:.2f} | ML Conf: {ml_prob:.2f}")
    print(f"News: {news_score:.2f} | Anomaly: {anom_score:.4f} | Gas: {gas_gwei:.1f} ({net_status})")

    # 5. The "Hybrid" Decision Logic
    if is_anomaly:
        print("🚨 CRITICAL: Market Crash Detected. HALTING.")
        final_action = "HALTED_ANOMALY"
        
    elif ml_pred == 1:
        # Check News AND Network
        if news_score < -0.2:
            print("Risk Manager: Blocked by Bad News.")
            final_action = "BLOCKED_NEWS"
        elif net_status == "LOW_ACTIVITY" and ml_prob < 0.7:
            # If network is dead, only buy if we are VERY confident
            print("Risk Manager: Network Dead. Requiring higher confidence.")
            final_action = "BLOCKED_LOW_VOL"
        else:
            size = 10000 * 0.02 
            execute_paper_trade(1, current_price, size, news_score, anom_score, gas_gwei)
            final_action = "BUY_EXECUTED"
    else:
        print("Model says HOLD/SELL.")
        final_action = "HOLD"

    # 6. Record to Memory
    log_system_state(current_time, current_price, news_score, ml_pred, ml_prob, final_action, anom_score, gas_gwei)
