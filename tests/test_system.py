import pytest
import os
import sys

# Add root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_loader import load_config

def test_config_structure():
    """Verify config.yaml loads and has the required sections."""
    config = load_config()
    assert 'trading' in config, "Config missing 'trading' section"
    assert 'risk' in config, "Config missing 'risk' section"
    assert config['trading']['symbol'] == 'BTC/USDT'
    assert config['system']['cycle_time'] > 0

def test_directory_structure():
    """Verify that essential project folders exist."""
    # Check if critical folders exist
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    assert os.path.exists(os.path.join(base_dir, 'data')), "Data folder missing"
    assert os.path.exists(os.path.join(base_dir, 'logs')), "Logs folder missing"
    assert os.path.exists(os.path.join(base_dir, 'models')), "Models folder missing"

if __name__ == "__main__":
    print("✅ Tests defined. Run with 'pytest tests/'")
