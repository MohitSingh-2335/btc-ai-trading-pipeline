import pandas as pd
import os
import sys

# Add path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# SETTINGS
RETRAIN_THRESHOLD = 50  # Retrain every 50 cycles (minutes)

def retrain_model():
    print("🔄 [MLOps] Starting Automated Retraining Sequence...")
    
    # 1. Load the "Memory" (New Data from the Loop)
    log_path = os.path.join('logs', 'system_memory.csv')
    
    if not os.path.exists(log_path):
        print("⚠️ No system logs found. Skipping.")
        return

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"⚠️ Could not read log file: {e}")
        return

    # 2. Check Threshold
    if len(df) < RETRAIN_THRESHOLD:
        print(f"⏳ Not enough new data ({len(df)}/{RETRAIN_THRESHOLD} rows). Waiting...")
        return

    print(f"✅ Triggering Retrain! Found {len(df)} new data points.")
    
    # 3. Archive the Memory (Don't delete it!)
    archive_path = os.path.join('logs', 'system_memory_archive.csv')
    
    # Append to archive
    if os.path.exists(archive_path):
         df.to_csv(archive_path, mode='a', header=False, index=False)
    else:
         df.to_csv(archive_path, index=False)
         
    # Clear the active log file for the next batch
    open(log_path, 'w').close() 

    # 4. RUN THE FULL LEARNING PIPELINE
    # This effectively "Re-Reads the whole textbook" including the new chapters
    
    print("   ⬇️  Fetching latest complete dataset...")
    exit_code = os.system("python3 scripts/ingest_clean.py > /dev/null")
    if exit_code != 0:
        print("❌ Error in Ingestion. Aborting.")
        return
    
    print("   🛠  Updating feature engineering...")
    exit_code = os.system("python3 scripts/feature_engine.py > /dev/null")
    if exit_code != 0:
        print("❌ Error in Feature Engineering. Aborting.")
        return
    
    print("   🧠  Training new LightGBM model...")
    exit_code = os.system("python3 scripts/train_model.py")
    
    if exit_code == 0:
        print("✅ Retraining Complete. Model Updated.")
    else:
        print("❌ Training Failed.")

if __name__ == "__main__":
    retrain_model()
