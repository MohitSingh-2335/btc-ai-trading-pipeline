import ccxt
import pandas as pd
import os
from datetime import datetime

def fetch_data(symbol='BTC/USDT', timeframe='1h', limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values(by='timestamp')
    
    cols = ['open', 'high', 'low', 'close', 'volume']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def save_data(df, filename):
    raw_path = os.path.join('data', 'raw', f"{filename}.parquet")
    clean_path = os.path.join('data', 'clean', f"{filename}.parquet")
    
    df.to_parquet(raw_path, index=False)
    df.to_parquet(clean_path, index=False)
    
    print(f"Saved {len(df)} rows to {clean_path}")

if __name__ == "__main__":
    print("Starting data ingestion...")
    df = fetch_data()
    df_clean = clean_data(df)
    save_data(df_clean, "btc_usdt_1h")
    print("Data ingestion complete.")
