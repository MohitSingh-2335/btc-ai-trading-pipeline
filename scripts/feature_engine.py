import pandas as pd
import talib
import numpy as np
import os

def load_data(filename):
    path = os.path.join('data', 'clean', f"{filename}.parquet")
    return pd.read_parquet(path)

def add_technical_indicators(df):
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
    return df

def add_on_chain_simulation(df):
    np.random.seed(42)
    volatility = df['close'].pct_change().rolling(window=24).std()
    
    df['active_addresses'] = (volatility * 1000000) + np.random.normal(0, 50000, len(df))
    df['active_addresses'] = df['active_addresses'].fillna(0).abs()
    
    df['network_hashrate_proxy'] = df['close'].rolling(window=168).mean() * np.random.uniform(0.95, 1.05, len(df))
    return df

def save_features(df, filename):
    path = os.path.join('features', f"{filename}_features.parquet")
    df = df.dropna()
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} feature rows to {path}")

if __name__ == "__main__":
    print("Starting feature engineering...")
    df = load_data("btc_usdt_1h")
    df = add_technical_indicators(df)
    df = add_on_chain_simulation(df)
    save_features(df, "btc_usdt_1h")
    print("Feature engineering complete.")
