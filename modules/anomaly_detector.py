import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(LSTMAutoencoder, self).__init__()
        # Encoder: Compresses data
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Decoder: Tries to reconstruct it
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

class MarketAnomalyDetector:
    def __init__(self):
        self.model = LSTMAutoencoder(input_dim=1)
        self.scaler = MinMaxScaler()
        self.threshold = 0.05  # Set a manual threshold for "High Error"
        
    def train_on_history(self, prices):
        """
        Quickly 'trains' the model on recent data to learn 'normal'.
        In a real app, you'd load a pre-trained file.
        """
        data = np.array(prices).reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        # Convert to Tensor
        X = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0) # Batch size 1
        
        # Simple training loop (Overfitting on purpose to learn 'current regime')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        self.model.train()
        for _ in range(50): # 50 Epochs
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()
            
    def detect_anomaly(self, recent_prices):
        """
        Returns: Anomaly Score (0 to 1), Is_Anomaly (Bool)
        """
        if len(recent_prices) < 30:
            return 0.0, False # Not enough data
            
        data = np.array(recent_prices).reshape(-1, 1)
        scaled_data = self.scaler.transform(data) # Use same scaler
        X = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(X)
            
        # Calculate Error (MSE)
        loss = torch.mean((X - reconstruction)**2).item()
        
        # If error is huge, it's an anomaly
        is_anomaly = loss > self.threshold
        return loss, is_anomaly

# Quick Test
if __name__ == "__main__":
    # Simulate normal prices
    prices = [100, 101, 102, 101, 100, 99, 100, 101]
    detector = MarketAnomalyDetector()
    detector.train_on_history(prices)
    
    # Test with a CRASH
    crash_prices = [100, 101, 102, 101, 100, 85, 80, 75] # Sudden drop
    score, is_anom = detector.detect_anomaly(crash_prices)
    
    print(f"Anomaly Score: {score:.4f}")
    print(f"Circuit Breaker Triggered? {is_anom}")
