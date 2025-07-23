import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from dotenv import load_dotenv
from app import config

# Load API Key from .env
load_dotenv()

def train_and_evaluate_model(df: pd.DataFrame):
    if df.empty:
        print("❌ Empty dataset. Training aborted.")
        return

    # Targets
    target_class = 'home_win'
    target_home_score = 'home_score'
    target_away_score = 'away_score'

    # ✅ Filter numeric features only
    feature_cols = [
        col for col in df.columns
        if col not in [target_class, target_home_score, target_away_score]
        and np.issubdtype(df[col].dtype, np.number)
    ]
    if not feature_cols:
        raise ValueError("No numeric features found in the dataset!")

    model_dir = os.path.join(config.NFL_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save feature columns
    joblib.dump(feature_cols, os.path.join(model_dir, 'model_features.pkl'))
    print(f"✅ Saved feature columns to: {os.path.join(model_dir, 'model_features.pkl')}")

    # Prepare X & y
    X = df[feature_cols]
    y_class = df[target_class]
    y_home = df[target_home_score]
    y_away = df[target_away_score]

    # ✅ Train/Test split — one call for all labels
    X_train, X_test, y_class_train, y_class_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_class, y_home, y_away, test_size=0.2, random_state=42
    )

    # ✅ Models
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    reg_home = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_away = RandomForestRegressor(n_estimators=100, random_state=42)

    # ✅ Train
    clf.fit(X_train, y_class_train)
    reg_home.fit(X_train, y_home_train)
    reg_away.fit(X_train, y_away_train)

    # ✅ Evaluate
    acc = accuracy_score(y_class_test, clf.predict(X_test))
    mse_home = mean_squared_error(y_home_test, reg_home.predict(X_test))
    mse_away = mean_squared_error(y_away_test, reg_away.predict(X_test))

    print(f"[🏈 home_win] Accuracy: {acc:.4f}")
    print(f"[🏠 home_score] MSE: {mse_home:.2f}")
    print(f"[🚗 away_score] MSE: {mse_away:.2f}")

    # ✅ Save models
    joblib.dump(clf, os.path.join(model_dir, 'model_home_win.pkl'))
    joblib.dump(reg_home, os.path.join(model_dir, 'model_home_score.pkl'))
    joblib.dump(reg_away, os.path.join(model_dir, 'model_away_score.pkl'))
    
    print(f"✅ Models saved to: {model_dir}")

    return clf, reg_home, reg_away


if __name__ == "__main__":
    csv_path = os.path.join(config.NFL_DIR, "historical_data_2010-2025.csv")
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        print("Please make sure the file 'head-to-head.csv' exists in the directory:", config.NFL_DIR)
    else:
        df = pd.read_csv(csv_path)
        train_and_evaluate_model(df)
