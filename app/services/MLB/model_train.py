from xgboost import XGBClassifier, XGBRegressor
from sklearn.calibration import CalibratedClassifierCV
from app.config import DATA_DIR
import joblib
import os
import pandas as pd

def train_historical_model():
    dir = os.path.join(DATA_DIR, "MLB")
    os.makedirs(dir, exist_ok=True)
    csv_file_path = os.path.join(dir, 'mlb_historical_data(2010-2024).csv')
    df= pd.read_csv(csv_file_path)
    print("df------\n", df)
     # Prepare inputs
    features = df.drop(['home_score', 'away_score', 'home_win'], axis=1)
    labels_home_win = df['home_win']
    labels_home_score = df['home_score']
    labels_away_score = df['away_score']
    # print("home_win value counts------\n", df['home_win'].value_counts())
    features = features.fillna(0)  # Handle missing values
    print( "training feature column----------", features.columns.tolist() )

    # Train XGBoost (with probability output)
     # --- Train Classification Model (Home Win Probability) ---
    clf_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
    clf_model.fit(features, labels_home_win)

    # Calibrate the classifier
    calibrated_model = CalibratedClassifierCV(estimator=clf_model, method='sigmoid', cv=3)
    calibrated_model.fit(features, labels_home_win)

    # --- Train Regression Models (Score Prediction) ---
    home_score_model = XGBRegressor(objective="reg:squarederror", random_state=42)
    away_score_model = XGBRegressor(objective="reg:squarederror", random_state=42)

    home_score_model.fit(features, labels_home_score)
    away_score_model.fit(features, labels_away_score)

    # --- Save Models ---
    model_dir = "models/MLB"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(calibrated_model, os.path.join(model_dir, 'mlb_win_classifier.pkl'))
    joblib.dump(home_score_model, os.path.join(model_dir, 'mlb_home_score_regressor.pkl'))
    joblib.dump(away_score_model, os.path.join(model_dir, 'mlb_away_score_regressor.pkl'))

    print(f"All models trained and saved to {model_dir}/")

    return clf_model, home_score_model, away_score_model

if __name__ == "__main__":
    # Train on multiple seasons for better generalization
    trained_model = train_historical_model() 