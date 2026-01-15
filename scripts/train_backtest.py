import pandas as pd
import lightgbm as lgb
import shap
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score

def load_features(filename):
    path = os.path.join('features', f"{filename}_features.parquet")
    return pd.read_parquet(path)

def create_target(df):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df.dropna()

def train_walk_forward(df):
    features = [c for c in df.columns if c not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
    X = df[features]
    y = df['target']
    
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    
    explainer = shap.TreeExplainer(model)
    
    return model, explainer, features

def save_model(model, explainer, filename):
    model_path = os.path.join('models', 'training', f"{filename}_model.joblib")
    explainer_path = os.path.join('models', 'training', f"{filename}_explainer.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    print("Starting ML training...")
    df = load_features("btc_usdt_1h")
    df = create_target(df)
    model, explainer, feats = train_walk_forward(df)
    save_model(model, explainer, "btc_usdt_1h")
    print("Training complete.")
