import pandas as pd
import joblib
import os
import json
import sys
import uuid
from datetime import datetime

# --- NEW IMPORTS (Standard & Custom) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.sentiment_analysis import SentimentPlugin
from modules.anomaly_detector import MarketAnomalyDetector
from modules.on_chain_data import OnChainPlugin
from utils.config_loader import load_config   # <--- Load Settings
from utils.logger import setup_logger         # <--- Professional Logging

# --- INITIALIZATION ---
config = load_config()       # Load config.yaml
logger = setup_logger()      # Start Logging System

def load_artifacts(filename):
    # Use paths from Config, not hardcoded strings
    model_path = config['paths']['model']
    features_path = config['paths']['data'].replace('raw', 'features').replace('.parquet', '_features.parquet')
    
    # Fallback if config path is generic, construct specific path
    if not os.path.exists(features_path):
         features_path = os.path.join('features', f"{filename}_features.parquet")

    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found at {model_path}")
        return None, None

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
    Saves snapshot including Web3 Data using Config paths
    """
    log_path = config['paths']['logs']
    
    new_row = {
        'timestamp': timestamp,
        'btc_price': price,
        'news_sentiment': sentiment,
        'model_prediction': prediction,
        'model_confidence': probability,
        'signal_type': signal_type,
        'anomaly_score': anomaly_score,
        'network_gas_gwei': gas_gwei
    }
    
    # Check header
    header = not os.path.exists(log_path)
    try:
        pd.DataFrame([new_row]).to_csv(log_path, mode='a', header=header, index=False)
        logger.info(f"📝 Memory Updated (Gas: {gas_gwei:.1f} Gwei)")
    except Exception as e:
        logger.error(f"❌ Failed to write logs: {e}")

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
        
    logger.info(f"✅ Trade EXECUTED: BUY ${size:.2f} BTC at ${price:.2f}")

if __name__ == "__main__":
    logger.info("--- Starting Multi-Modal AI Cycle ---")
    
    # 1. Technical (LightGBM)
    model, df = load_artifacts("btc_usdt_1h")
    if model is None: sys.exit()
    
    ml_pred, ml_prob, current_price = get_latest_signal(model, df)
    
    # 2. Sentiment (NLP) - WRAPPED IN TRY/EXCEPT (Safe Mode)
    try:
        sentiment_plugin = SentimentPlugin()
        # Capture both signal and score
        news_signal, news_score = sentiment_plugin.get_sentiment_signal()
    except Exception as e:
        logger.warning(f"⚠️ NLP Error: {e} (Defaulting to Neutral)")
        news_signal, news_score = 0, 0.0
    
    # 3. Risk (LSTM)
    detector = MarketAnomalyDetector()
    recent_prices = df['close'].values[-100:]
    detector.train_on_history(recent_prices)
    anom_score, is_anomaly = detector.detect_anomaly(df['close'].values[-30:])
    
    # 4. On-Chain (Web3)
    web3_plugin = OnChainPlugin()
    gas_gwei, net_status = web3_plugin.get_network_health()
    
    current_time = datetime.now().isoformat()
    final_action = "HOLD"

    logger.info(f"Price: ${current_price:.2f} | Conf: {ml_prob:.2f} | News: {news_score:.2f} | Gas: {gas_gwei:.1f}")

    # 5. The "Hybrid" Decision Logic (USING CONFIG VALUES)
    if is_anomaly:
        logger.critical("🚨 CRITICAL: Market Crash Detected. HALTING.")
        final_action = "HALTED_ANOMALY"
        
    elif ml_pred == 1:
        # Check News AND Network using CONFIG thresholds
        if news_score < config['risk']['sentiment_threshold']:
            logger.info(f"Risk Manager: Blocked by Bad News ({news_score:.2f} < {config['risk']['sentiment_threshold']})")
            final_action = "BLOCKED_NEWS"
            
        elif gas_gwei > config['risk']['max_gas_gwei']:
             logger.info(f"Risk Manager: Blocked by High Gas ({gas_gwei} > {config['risk']['max_gas_gwei']})")
             final_action = "BLOCKED_GAS"

        elif ml_prob < config['trading']['min_confidence']:
            logger.info(f"Risk Manager: Confidence too low ({ml_prob:.2f} < {config['trading']['min_confidence']})")
            final_action = "BLOCKED_LOW_CONF"
            
        else:
            # Calculate Position Size (e.g., 2% of $10,000)
            # In a real bot, you'd fetch your wallet balance here.
            # For now, we assume a $10,000 portfolio.
            portfolio_value = 10000 
            size = portfolio_value * config['trading']['trade_size_pct']
            
            execute_paper_trade(1, current_price, size, news_score, anom_score, gas_gwei)
            final_action = "BUY_EXECUTED"
    else:
        logger.info("Model says HOLD/SELL.")
        final_action = "HOLD"

    # 6. Record to Memory
    log_system_state(current_time, current_price, news_score, ml_pred, ml_prob, final_action, anom_score, gas_gwei)
