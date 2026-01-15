import pandas as pd
import joblib
import os
import json
import sys
import uuid
from datetime import datetime

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.sentiment_analysis import SentimentPlugin

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

def execute_paper_trade(signal, price, size, sentiment_score):
    trade_id = str(uuid.uuid4())
    trade_data = {
        "id": trade_id,
        "timestamp": datetime.now().isoformat(),
        "action": "BUY" if signal == 1 else "SELL",
        "price": price,
        "size_usdt": size,
        "sentiment_score": sentiment_score,  # Log the news sentiment too
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
        
    print(f"Trade EXECUTED: BUY ${size:.2f} BTC at ${price:.2f} | Sentiment: {sentiment_score:.2f}")

if __name__ == "__main__":
    print("Starting trading cycle...")
    
    # 1. Load ML Model
    model, df = load_artifacts("btc_usdt_1h")
    ml_pred, ml_prob, current_price = get_latest_signal(model, df)
    
    # 2. Get News Sentiment
    sentiment_plugin = SentimentPlugin()
    news_signal, news_score = sentiment_plugin.get_sentiment_signal()
    
    print(f"Current Price: ${current_price}")
    print(f"ML Prediction: {'UP' if ml_pred == 1 else 'DOWN'} (Conf: {ml_prob:.2f})")
    print(f"News Sentiment: {news_score:.2f} ({'BULLISH' if news_signal == 1 else 'BEARISH' if news_signal == -1 else 'NEUTRAL'})")
    
    # 3. Decision Logic (The "Hedge Fund" Rule)
    if ml_pred == 1:
        if news_signal == -1:
            print("Risk Manager: ML says BUY, but News is NEGATIVE. Trade BLOCKED.")
        else:
            # Buy if ML says UP and News is NOT Negative
            size = 10000 * 0.02 # 2% risk
            execute_paper_trade(1, current_price, size, news_score)
    else:
        print("Risk Manager: No trade entry conditions met.")
