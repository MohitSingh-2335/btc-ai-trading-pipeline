import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    print("🧠 Loading Features...")
    feature_path = os.path.join('features', 'btc_usdt_1h_features.parquet')
    
    if not os.path.exists(feature_path):
        print("❌ Error: Feature file not found. Run feature_engine.py first.")
        return

    df = pd.read_parquet(feature_path)

    # Define predictors (Everything except non-feature columns)
    non_features = ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']
    predictors = [c for c in df.columns if c not in non_features]

    X = df[predictors]
    y = df['target']

    # Split Data (80% Train, 20% Test)
    # shuffle=False is crucial for Time Series (can't learn from the future)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"🏋️ Training LightGBM Model on {len(X_train)} rows...")

    # --- THE FIX: Stricter Hyperparameters ---
    model = lgb.LGBMClassifier(
        n_estimators=100,        # Fewer trees (prevents overfitting)
        learning_rate=0.05,      # Slower learning
        max_depth=3,             # Shallow trees (forces generalization)
        num_leaves=8,            # Simpler decision boundaries
        reg_alpha=0.1,           # L1 Regularization
        reg_lambda=0.1,          # L2 Regularization
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Evaluation:")
    print(f"   Accuracy: {acc:.2%}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the new brain
    save_path = os.path.join('models', 'training', 'btc_usdt_1h_model.joblib')
    joblib.dump(model, save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
