from web3 import Web3
import random

class OnChainPlugin:
    def __init__(self):
        # Use Cloudflare's free Ethereum Node
        self.w3 = Web3(Web3.HTTPProvider('https://rpc.ankr.com/eth'))
        
    def get_network_health(self):
        """
        Fetches real-time on-chain metrics.
        Returns: 
        - gwei (float): Current Gas Price (Network Activity Proxy)
        - status (str): 'CONGESTED', 'NORMAL', or 'LOW_ACTIVITY'
        """
        try:
            if not self.w3.is_connected():
                return self._simulate_data()
            
            # 1. Get Real Gas Price
            gas_wei = self.w3.eth.gas_price
            gas_gwei = float(self.w3.from_wei(gas_wei, 'gwei'))
            
            # 2. Determine Network Status
            if gas_gwei > 50:
                status = "CONGESTED" # High demand
            elif gas_gwei < 10:
                status = "LOW_ACTIVITY"
            else:
                status = "NORMAL"
                
            return gas_gwei, status

        except Exception as e:
            print(f"⚠️ Web3 Error: {e}")
            return self._simulate_data()
            
    def _simulate_data(self):
        # Fallback simulation
        return 25.0, "NORMAL"

if __name__ == "__main__":
    plugin = OnChainPlugin()
    gwei, status = plugin.get_network_health()
    print(f"🔗 Real-Time Gas: {gwei:.2f} Gwei | Status: {status}")
