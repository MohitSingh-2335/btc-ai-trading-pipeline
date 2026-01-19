import time
import os

print("🚀 Starting AI Trading Loop (Self-Healing Enabled)...")
print("Press Ctrl+C to stop.")

cycle_count = 0

try:
    while True:
        cycle_count += 1
        print(f"\n--- Cycle {cycle_count} ---")
        
        print("[1/3] Fetching & Engineering...")
        os.system("python3 scripts/ingest_clean.py")
        os.system("python3 scripts/feature_engine.py")
        
        print("[2/3] AI Analysis (LightGBM + LSTM)...")
        os.system("python3 scripts/risk_trade.py")
        
        # Every 10 cycles, check if we need to retrain
        if cycle_count % 10 == 0:
            print("[MLOps] Checking for Retraining opportunity...")
            os.system("python3 scripts/auto_retrain.py")
        
        print("✅ Sleeping for 60 seconds...")
        time.sleep(60)

except KeyboardInterrupt:
    print("\n🛑 Loop stopped by user.")
