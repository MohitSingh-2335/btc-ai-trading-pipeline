import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

class SentimentPlugin:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze_headlines(self, headlines):
        """
        Input: List of strings (headlines)
        Output: Average compound score (-1 to 1)
        """
        scores = [self.sia.polarity_scores(h)['compound'] for h in headlines]
        return np.mean(scores) if scores else 0

    def get_sentiment_signal(self):
        # In a real app, use requests.get('https://cryptopanic.com/api/...')
        # For this Resume Demo, we simulate realistic headlines
        
        simulated_news = [
            "Bitcoin breaks resistance levels as institutional demand grows",
            "SEC delays decision on crypto ETF, market uncertain",
            "Miners are holding more BTC than ever before",
            "Tech stocks tumble, dragging crypto down slightly"
        ]
        
        score = self.analyze_headlines(simulated_news)
        
        # Logic: If news is very positive (> 0.05), return BULLISH signal (1)
        if score > 0.05:
            return 1, score
        elif score < -0.05:
            return -1, score
        else:
            return 0, score

# Quick test when running the file directly
if __name__ == "__main__":
    plugin = SentimentPlugin()
    signal, score = plugin.get_sentiment_signal()
    print(f"Sentiment Score: {score:.4f}")
    print(f"Signal: {'BUY' if signal == 1 else 'SELL' if signal == -1 else 'NEUTRAL'}")
