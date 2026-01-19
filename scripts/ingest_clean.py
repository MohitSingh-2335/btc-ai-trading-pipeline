import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_data(symbol='BTC/USDT', timeframe='1h', limit=10000):
    print(f"📥 Starting Batch Download for {symbol} ({limit} candles)...")
    
    # 1. Setup Binance
    exchange = ccxt.binance({
        'enableRateLimit': True, # Auto-throttle requests to avoid bans
        'options': {'defaultType': 'future'} # Use Futures market for better volume data
    })
    
    # 2. Calculate Start Time (Go back 'limit' hours)
    # 1h = 3600 seconds * 1000 ms
    duration_ms = limit * 60 * 60 * 1000
    since = exchange.milliseconds() - duration_ms
    
    all_candles = []
    
    while len(all_candles) < limit:
        try:
            # Fetch batch (Binance max is usually 1000-1500 per call)
            fetch_limit = 1000 
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=fetch_limit)
            
            if not candles:
                print("⚠️ No more data available.")
                break
                
            # Append to list
            all_candles += candles
            
            # Update 'since' to the timestamp of the last candle + 1ms
            last_timestamp = candles[-1][0]
            since = last_timestamp + 1
            
            # print(f"   ✅ Fetched {len(candles)} candles... (Total: {len(all_candles)})")
            
            # Small sleep to be safe
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            break

    # 3. Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 4. Clean Data
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp')
    
    # Keep only the requested amount (in case we fetched a bit too much)
    df = df.tail(limit)

    # 5. Save
    # We save to BOTH raw and clean folders to keep the rest of your pipeline happy
    # (Your old code did this, so we should keep that structure)
    
    raw_dir = os.path.join('data', 'raw')
    clean_dir = os.path.join('data', 'clean')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    raw_path = os.path.join(raw_dir, 'btc_usdt_1h.parquet')
    clean_path = os.path.join(clean_dir, 'btc_usdt_1h.parquet')
    
    df.to_parquet(raw_path, index=False)
    df.to_parquet(clean_path, index=False)
    
    print(f"✅ Ingestion Complete: {len(df)} rows ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
if __name__ == "__main__":
    fetch_data()
