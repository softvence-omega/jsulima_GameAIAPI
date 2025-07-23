from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.config import DATA_DIR
import joblib
import os
import pandas as pd

def train_model():
    csv_dir = os.path.join(DATA_DIR, 'csv', "MLB")
    os.makedirs(csv_dir, exist_ok=True)
    csv_file_path = os.path.join(csv_dir, 'batter_stats_data(2010-2024).csv')
    df= pd.read_csv(csv_file_path)

     # Prepare inputs
    drop_cols = ['hits', 'home_runs', 'rbis', 'game_id', 'player_id', 'team_id', 'opponent_team_id',"player_name",]
    features = df.drop(columns=drop_cols)
    features = features.fillna(0)

    # Encode categorical variables if any
    cat_cols = features.select_dtypes(include=['object']).columns
    features = pd.get_dummies(features, columns=cat_cols)

    # Labels create
    labels_hits = df['hits']
    labels_home_runs = df['home_runs']
    labels_rbis = df['rbis']

    # train test split
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels_hits, test_size=0.2, random_state=42)
    features = features.fillna(0)  # Handle missing values
    print( "training feature column----------", features.columns.tolist() )

    # Initialize models
    hits_model = XGBRegressor(objective="reg:squarederror", random_state=42)
    home_runs_model = XGBRegressor(objective="reg:squarederror", random_state=42)
    rbis_model = XGBRegressor(objective="reg:squarederror", random_state=42)

    # Train models
    hits_model.fit(features_train, labels_train)
    home_runs_model.fit(features, labels_home_runs)
    rbis_model.fit(features, labels_rbis)

      # --- Evaluate Models ---
    hits_pred = hits_model.predict(features_test)
    print("hits Model Evaluation:")
    # print(f"  RMSE: {mean_squared_error(labels_test, strikeouts_score, squared=False):.2f}")
    print(f"  MAE: {mean_absolute_error(labels_test, hits_pred):.2f}")
    print(f"  R²: {r2_score(labels_test, hits_pred):.4f}")

    # Save models
    model_dir = "models/MLB/batters"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(hits_model, os.path.join(model_dir, 'hits_regressor.pkl'))
    joblib.dump(home_runs_model, os.path.join(model_dir, 'home_runs_regressor.pkl'))
    joblib.dump(rbis_model, os.path.join(model_dir, 'rbis_regressor.pkl'))

    print(f"All models trained and saved to {model_dir}/")

    return hits_model, home_runs_model, rbis_model

if __name__ == "__main__":
    # Train on multiple seasons for better generalization
    trained_model = train_model() 
