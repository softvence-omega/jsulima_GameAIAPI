# predictor.py

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib

class PitcherPredictor:
    def __init__(self, csv_path="app/data/MLB/pitching/pitching_data_combined.csv"):
        self.csv_path = csv_path
        self.model = None
        self.final_output = None
        self._load_and_train()

    def _load_and_train(self):
        df = pd.read_csv(self.csv_path)

        # Ensure numeric types for aggregation columns
        numeric_cols = [
            'innings_pitched', 'strikeouts', 'earned_runs', 'walks', 'hits',
            'home_runs', 'hbp', 'earned_runs_average', 'pc_st'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with all-NaN in numeric columns (optional, for safety)
        df = df.dropna(subset=numeric_cols, how='all')

        # Aggregate stats
        agg_df = df.groupby(['team', 'name']).agg({
            'innings_pitched': 'sum',
            'strikeouts': 'sum',
            'earned_runs': 'sum',
            'walks': 'sum',
            'hits': 'sum',
            'home_runs': 'sum',
            'hbp': 'sum',
            'earned_runs_average': 'mean',
            'pc_st': 'sum'
        }).reset_index()

        # Remove rows with NaN or inf in features
        features = [
            'innings_pitched', 'strikeouts', 'earned_runs', 'walks', 'hits',
            'home_runs', 'hbp', 'earned_runs_average', 'pc_st'
        ]
        # Ensure all features are numeric after aggregation
        for col in features:
            agg_df[col] = pd.to_numeric(agg_df[col], errors='coerce')
        agg_df = agg_df.replace([float('inf'), float('-inf')], pd.NA)
        agg_df = agg_df.dropna(subset=features)
        # Force all features to float
        agg_df[features] = agg_df[features].astype(float)
        # Debug: print dtypes to ensure all are float

        # Synthetic confidence score
        agg_df['confidence_score'] = (
            agg_df['strikeouts'] * 2
            - agg_df['earned_runs'] * 3
            - agg_df['walks'] * 1.5
            - agg_df['hits']
            + agg_df['innings_pitched'] * 2
        )

        # Performance score (example formula)
        agg_df['performance_score'] = (
            agg_df['strikeouts'] + agg_df['innings_pitched']
            - agg_df['earned_runs'] - agg_df['walks']
            - agg_df['hits'] - agg_df['home_runs'] - agg_df['hbp']
        )

        # Scale performance_score and confidence_score to 0-100
        def minmax_scale(series):
            if series.max() == series.min():
                return 100 * (series - series.min())  # all same value, set to 0
            return 100 * (series - series.min()) / (series.max() - series.min())
        agg_df['performance_score_scaled'] = minmax_scale(agg_df['performance_score'])
        agg_df['confidence_score_scaled'] = minmax_scale(agg_df['confidence_score'])

        target = 'confidence_score'
        X = agg_df[features]
        y = agg_df[target]

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = XGBRegressor(n_estimators=25, learning_rate=0.1, random_state=42)
        self.model.fit(X_train, y_train)
        self.model.save_model('models/mlb_pitcher_model.pkl')

        # Predict and rank
        agg_df['predicted_score'] = self.model.predict(X)
        top_pitchers = agg_df.sort_values('predicted_score', ascending=False).groupby('team').head(1)

        # Final output
        self.final_output = top_pitchers[['team', 'name', 'innings_pitched', 'strikeouts', 'earned_runs', 'predicted_score', 'performance_score', 'performance_score_scaled', 'confidence_score_scaled']]
        self.final_output = self.final_output.rename(columns={
            'team': 'team_name',
            'name': 'player_name',
            'predicted_score': 'confidence_score'
        })

    def get_top_pitcher(self, home_team_name: str, away_team_name: str):
        if self.final_output is None:
            return None
        home_result = self.final_output[self.final_output['team_name'] == home_team_name]
        away_result = self.final_output[self.final_output['team_name'] == away_team_name]
        result = {}
        if isinstance(home_result, pd.DataFrame) and not home_result.empty:
            pitcher = home_result.iloc[0].to_dict()
            result['home_team_pitcher'] = pitcher
        if isinstance(away_result, pd.DataFrame) and not away_result.empty:
            pitcher = away_result.iloc[0].to_dict()
            result['away_team_pitcher'] = pitcher
        if not result:
            return None
        return result



if __name__ == "__main__":
    predictor = PitcherPredictor()
    print(predictor.final_output.head())
    # Example usage
    top_pitchers = predictor.get_top_pitcher("Chicago Cubs", "Baltimore Orioles")
    print(top_pitchers)