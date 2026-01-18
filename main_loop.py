import time
import os

print("🚀 Starting AI Trading Loop...")
print("Press Ctrl+C to stop.")

try:
    while True:
        print("\n[1/3] Fetching latest Binance data...")
        os.system("python3 scripts/ingest_clean.py")
        
        print("[2/3] Engineering features...")
        os.system("python3 scripts/feature_engine.py")
        
        print("[3/3] AI Model analyzing & Trading...")
        os.system("python3 scripts/risk_trade.py")
        
        print("✅ Cycle complete. Sleeping for 60 seconds...")
        time.sleep(60)

except KeyboardInterrupt:
    print("\n🛑 Loop stopped by user.")
