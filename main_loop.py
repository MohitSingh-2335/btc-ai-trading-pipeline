import time
import os
from datetime import datetime

# ANSI Colors for "Hacker Mode" text
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

def print_header(cycle):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{CYAN}{'='*60}")
    print(f"🔄 CYCLE {cycle:04d} | 🕒 {now}")
    print(f"{'='*60}{RESET}")

if __name__ == "__main__":
    print(f"{GREEN}🚀 Starting AI Trading System (Quiet Mode)...{RESET}")
    print("Press Ctrl+C to stop.")

    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            print_header(cycle_count)
            
            # Step 1: Data
            print(f"⬇️  {YELLOW}Fetching Market Data...{RESET}", end=" ")
            os.system("python3 scripts/ingest_clean.py > /dev/null") # > /dev/null silences the output completely
            print(f"{GREEN}[DONE]{RESET}")

            # Step 2: Features
            print(f"🛠  {YELLOW}Engineering Features...{RESET}", end=" ")
            os.system("python3 scripts/feature_engine.py > /dev/null")
            print(f"{GREEN}[DONE]{RESET}")
            
            # Step 3: Analysis & Trade (We keep this output visible because it's the result)
            print(f"🧠  {YELLOW}Analyzing & Trading...{RESET}\n")
            print(f"{'-'*60}")
            os.system("python3 scripts/risk_trade.py")
            print(f"{'-'*60}")
            
            # Step 4: MLOps
            if cycle_count % 10 == 0:
                print(f"🤖  {CYAN}[MLOps] Checking Auto-Retrain...{RESET}")
                os.system("python3 scripts/auto_retrain.py")
            
            print(f"\n💤 Sleeping 60s...")
            time.sleep(60)

    except KeyboardInterrupt:
        print(f"\n{RED}🛑 System Halted by User.{RESET}")
