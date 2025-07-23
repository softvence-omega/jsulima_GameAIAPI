import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from app import config

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ File not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)

def prepare_classification_features(df: pd.DataFrame):
    y = df['home_win']
    df = df.drop(columns=['home_win', 'home_score', 'away_score'], errors='ignore')
    X = df.select_dtypes(include='number')
    return X, y

def prepare_regression_features(df: pd.DataFrame):
    y_home = df['home_score']
    y_away = df['away_score']
    df = df.drop(columns=['home_win', 'home_score', 'away_score'], errors='ignore')
    X = df.select_dtypes(include='number')
    return X, y_home, y_away

def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.name == 'home_win' else None)

def train_classifier(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_regressors(X_train, y_home_train, y_away_train):
    home_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
    home_regressor.fit(X_train, y_home_train)

    away_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
    away_regressor.fit(X_train, y_away_train)

    return home_regressor, away_regressor

def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🏆 Accuracy: {acc:.2%}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"💾 Model saved successfully: {filename}")

def save_feature_columns(feature_cols, filename):
    joblib.dump(feature_cols, filename)
    print(f"💾 Feature columns saved successfully: {filename}")

def train_and_evaluate(csv_path):
    df = load_data(csv_path)

    # Classification pipeline
    X_cls, y_cls = prepare_classification_features(df)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = split_data(X_cls, y_cls)
    classifier = train_classifier(X_train_cls, y_train_cls)
    evaluate_classifier(classifier, X_test_cls, y_test_cls)

    # Regression pipeline
    X_reg, y_home, y_away = prepare_regression_features(df)
    X_train_reg, X_test_reg, y_home_train, y_home_test = split_data(X_reg, y_home)
    _, _, y_away_train, y_away_test = split_data(X_reg, y_away)

    home_regressor, away_regressor = train_regressors(X_train_reg, y_home_train, y_away_train)

    # Save all
    save_model(classifier, os.path.join(config.NFL_MODEL_DIR, 'nfl_home_win_classifier.pkl'))
    save_model(home_regressor, os.path.join(config.NFL_MODEL_DIR, 'nfl_home_score_regressor.pkl'))
    save_model(away_regressor, os.path.join(config.NFL_MODEL_DIR, 'nfl_away_score_regressor.pkl'))

    feature_cols = list(X_cls.columns)
    save_feature_columns(feature_cols, os.path.join(config.NFL_MODEL_DIR, 'feature_columns.pkl'))

if __name__ == "__main__":
    csv_path = os.path.join(config.NFL_DIR, "historical_data_2010-2025.csv")
    try:
        train_and_evaluate(csv_path)
    except Exception as e:
        print(f"❌ Error: {e}")
