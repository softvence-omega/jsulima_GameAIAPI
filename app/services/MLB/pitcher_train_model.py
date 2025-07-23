from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.config import DATA_DIR
import pandas as pd
from app.services.MLB.playerDataProcessor import PlayerDataProcessor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from dotenv import load_dotenv
load_dotenv()

# processor= BaseballDataProcessor(os.getenv('GOALSERVE_API_KEY'))
p_processor = PlayerDataProcessor()

def train_model():
    # df = processor.get_pitcher_data()

    csv_dir = os.path.join(DATA_DIR, "MLB")
    os.makedirs(csv_dir, exist_ok=True)
    csv_file_path = os.path.join(csv_dir, 'pitcher_stats_data(2010-2024).csv')
    df= pd.read_csv(csv_file_path)
    print("columns in df------------------", df.columns)

    df = p_processor.create_pitcher_features(df)
    feature_cols = p_processor.get_pitcher_feature_columns()
    
    # Prepare data
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # # Scale features
    # p_processor.scalers['pitcher'] = StandardScaler()
    # X_scaled = p_processor.scalers['pitcher'].fit_transform(X)
    # X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    labels_innings_pitched = df['innings_pitched']
    labels_strikeouts = df['strikeouts']
    labels_earned_runs = df['earned_runs']

    
    # train test split
    features_train, features_test, labels_train, labels_test = train_test_split(X, labels_strikeouts, test_size=0.2, random_state=42)
    # X_scaled = X_scaled.fillna(0)  # Handle missing values
    print( "training feature column----------", X.columns.tolist() )

    # --- Train Regression Models (Score Prediction) ---
    innings_pitched_model = RandomForestRegressor(random_state=42)
    strikeouts_model = RandomForestRegressor(random_state=42)
    earned_runs_model = RandomForestRegressor(random_state=42)

    innings_pitched_model.fit(features_train, labels_train)
    strikeouts_model.fit(X, labels_strikeouts)
    earned_runs_model.fit(X, labels_earned_runs)

    # --- Evaluate Models ---
    strikeouts_score = strikeouts_model.predict(features_test)
    print("Strikeouts Model Evaluation:")
    # print(f"  RMSE: {mean_squared_error(labels_test, strikeouts_score, squared=False):.2f}")
    print(f"  MAE: {mean_absolute_error(labels_test, strikeouts_score):.2f}")
    print(f"  R²: {r2_score(labels_test, strikeouts_score):.4f}")


    # --- Save Models ---
    model_dir = "models/MLB/pitcher"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(innings_pitched_model, os.path.join(model_dir, 'mlb_innings_pitched_regressor.pkl'))
    joblib.dump(strikeouts_model, os.path.join(model_dir, 'mlb_strikeouts_regressor.pkl'))
    joblib.dump(earned_runs_model, os.path.join(model_dir, 'mlb_earned_runs_regressor.pkl'))

    print(f"All models trained and saved to {model_dir}/")

    return innings_pitched_model, strikeouts_model, earned_runs_model

if __name__ == "__main__":
    # Train on multiple seasons for better generalization
    trained_model = train_model() 