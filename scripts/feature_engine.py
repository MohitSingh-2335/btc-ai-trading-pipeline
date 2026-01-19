import pandas as pd
import talib
import numpy as np
import os

def load_data(filename):
    path = os.path.join('data', 'clean', f"{filename}.parquet")
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found.")
        return pd.DataFrame()
    return pd.read_parquet(path)

def add_technical_indicators(df):
    # Your professional TA-Lib indicators
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
    return df

def add_on_chain_simulation(df):
    # Keep your simulation logic (it helps the model learn noise)
    np.random.seed(42)
    volatility = df['close'].pct_change().rolling(window=24).std()
    
    df['active_addresses'] = (volatility * 1000000) + np.random.normal(0, 50000, len(df))
    df['active_addresses'] = df['active_addresses'].fillna(0).abs()
    
    df['network_hashrate_proxy'] = df['close'].rolling(window=168).mean() * np.random.uniform(0.95, 1.05, len(df))
    return df

def create_target(df):
    # --- THIS IS THE MISSING PIECE ---
    # 1 = Price goes UP next hour, 0 = Price goes DOWN
    # We shift(-1) to look into the future
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df

def save_features(df, filename):
    path = os.path.join('features', f"{filename}_features.parquet")
    
    # Drop rows with NaN (caused by indicators warming up)
    df = df.dropna()
    
    df.to_parquet(path, index=False)
    print(f"✅ Saved {len(df)} feature rows to {path}")

if __name__ == "__main__":
    print("Starting feature engineering...")
    df = load_data("btc_usdt_1h")
    
    if not df.empty:
        df = add_technical_indicators(df)
        df = add_on_chain_simulation(df)
        df = create_target(df)  # <--- NEW STEP
        save_features(df, "btc_usdt_1h")
        print("Feature engineering complete.")
