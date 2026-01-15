import pandas as pd
import joblib
import os
import json
from datetime import datetime
import uuid

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

def calculate_position_size(account_balance, risk_per_trade=0.02):
    return account_balance * risk_per_trade

def execute_paper_trade(signal, price, size):
    if signal == 1:
        trade_id = str(uuid.uuid4())
        trade_data = {
            "id": trade_id,
            "timestamp": datetime.now().isoformat(),
            "action": "BUY",
            "price": price,
            "size_usdt": size,
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
            
        print(f"Trade EXECUTED: BUY ${size:.2f} BTC at ${price:.2f} (ID: {trade_id})")
    else:
        print("Signal is SELL/HOLD. No trade executed.")

if __name__ == "__main__":
    print("Starting trading cycle...")
    model, df = load_artifacts("btc_usdt_1h")
    pred, prob, current_price = get_latest_signal(model, df)
    
    print(f"Current Price: ${current_price}")
    print(f"Model Prediction: {'UP' if pred == 1 else 'DOWN'} (Confidence: {prob:.2f})")
    
    if pred == 1:
        size = calculate_position_size(10000) 
        execute_paper_trade(pred, current_price, size)
    else:
        print("Risk Manager: No trade entry conditions met.")
