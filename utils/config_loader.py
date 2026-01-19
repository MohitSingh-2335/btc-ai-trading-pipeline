import yaml
import os

def load_config():
    """Loads the config.yaml file from the root directory."""
    # Find the config.yaml file (assuming it's in the root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found at {config_path}")

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Test it
if __name__ == "__main__":
    conf = load_config()
    print(f"✅ Config Loaded! Trading Symbol: {conf['trading']['symbol']}")
