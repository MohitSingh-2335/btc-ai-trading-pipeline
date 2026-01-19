import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime

class SentimentPlugin:
    def __init__(self):
        # We use CoinTelegraph's RSS Feed (Free & Real-time)
        self.rss_url = "https://cointelegraph.com/rss"
        self.analyzer = SentimentIntensityAnalyzer()

    def fetch_headlines(self):
        """Fetches the latest 20 headlines from CoinTelegraph."""
        try:
            feed = feedparser.parse(self.rss_url)
            headlines = []
            
            # Get top 20 items
            for entry in feed.entries[:20]:
                headlines.append(entry.title)
                
            return headlines
        except Exception as e:
            print(f"⚠️ NLP Error: {e}")
            return []

    def get_sentiment_signal(self):
        """
        Analyzes headlines and returns:
        - Signal: 1 (Buy), -1 (Sell), 0 (Neutral)
        - Score: Float from -1.0 to 1.0 (The raw average sentiment)
        """
        headlines = self.fetch_headlines()
        
        if not headlines:
            return 0, 0.0 # Default if internet is down

        total_score = 0
        count = 0

        # Analyze each headline
        for title in headlines:
            # VADER gives a 'compound' score from -1 (Negative) to +1 (Positive)
            score = self.analyzer.polarity_scores(title)['compound']
            total_score += score
            count += 1
            
        # Average the sentiment
        if count > 0:
            avg_score = total_score / count
        else:
            avg_score = 0.0

        # Determine Signal based on Thresholds
        # If sentiment is > 0.05, it's Bullish. < -0.05 is Bearish.
        if avg_score > 0.05:
            signal = 1
        elif avg_score < -0.05:
            signal = -1
        else:
            signal = 0
            
        return signal, avg_score

# Quick Test to verify it works
if __name__ == "__main__":
    bot = SentimentPlugin()
    sig, score = bot.get_sentiment_signal()
    
    print(f"\n📰 Real-Time Crypto News Analysis")
    print(f"--------------------------------")
    print(f"📉 Sentiment Score: {score:.4f}")
    print(f"🚦 Signal: {'BULLISH' if sig == 1 else 'BEARISH' if sig == -1 else 'NEUTRAL'}")
    
    print("\n🔍 Latest Headlines Analyzed:")
    headlines = bot.fetch_headlines()
    for i, h in enumerate(headlines[:5]):
        print(f"   {i+1}. {h}")
