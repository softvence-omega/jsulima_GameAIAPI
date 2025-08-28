import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, accuracy_score
import joblib
import os
import requests
from app.config import IMAGE_URL


from app.api.v1.endpoints.nfl_lineup_prediction import get_injured_players
from app.config import (
    GOALSERVE_BASE_URL, GOALSERVE_API_KEY
)


class MLBLineupPredictionService:
    def __init__(self, team_name: str):
        self.team_name = team_name
        self.bat_model_path = "models/mlb_batting_model.pkl"
        self.pos_model_path = "models/mlb_fielding_model.pkl"

        self.df_rosters, self.df_stats, self.df_inj = self._load_data()
        self._preprocess_data()  # Centralized preprocessing for all data
        self.rf_bat, self.rf_pos = self._load_or_train_models()


    def _load_data(self):
        file_paths = {
            "rosters": "app/data/MLB/mlb_rosters.csv",
            "player_stats": "app/data/MLB/mlb_player_stats.csv",
            "injuries": "app/data/MLB/mlb_injuries.csv"
        }
        dfs = {name: pd.read_csv(path) for name, path in file_paths.items()}
        return dfs["rosters"], dfs["player_stats"], dfs["injuries"]

    def _height_to_inches(self, ht):
        try:
            feet, inches = str(ht).split("'")
            inches = inches.replace('"', '').strip()
            return int(feet) * 12 + int(inches)
        except (ValueError, AttributeError):
            return np.nan

    def _preprocess_data(self):
        """Central method to preprocess all dataframes after loading."""
        # Process rosters for height and weight
        self.df_rosters["height_in"] = self.df_rosters["height"].apply(self._height_to_inches)
        self.df_rosters["weight_int"] = self.df_rosters["weight"].str.replace("lbs", "",
                                                                              regex=False).str.strip().astype("float")

        # Process stats to create training labels
        df_stats = self.df_stats.copy()
        df_stats["ops_proxy"] = df_stats["on_base_percentage"].fillna(0) + df_stats["slugging_percentage"].fillna(0)
        df_stats["bat_order_label"] = (
                df_stats
                .sort_values(["team", "ops_proxy"], ascending=[True, False])
                .groupby("team")
                .cumcount() + 1
        )
        df_stats.loc[df_stats["bat_order_label"] > 9, "bat_order_label"] = 9
        self.df_stats = df_stats

    def _train_batting_model(self):
        bat_feat_cols = [
            "on_base_percentage", "slugging_percentage", "batting_avg",
            "home_runs", "doubles", "triples", "stolen_bases", "strikeouts"
        ]
        X_bat = self.df_stats[bat_feat_cols]
        y_bat = self.df_stats["bat_order_label"]

        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        preprocess_bat = ColumnTransformer(transformers=[("num", numeric_transformer, bat_feat_cols)])

        rf_bat = Pipeline(steps=[
            ("preprocess", preprocess_bat),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
        ])
        rf_bat.fit(X_bat, y_bat)

        joblib.dump(rf_bat, self.bat_model_path)
        return rf_bat

    def _train_fielding_model(self):
        df_rosters = self.df_rosters.copy()  # Use preprocessed data

        field_feat_cols = ["bats", "throws", "age", "height_in", "weight_int"]
        X_pos = df_rosters[field_feat_cols].copy()
        y_pos = df_rosters["player_position"].copy()

        # Handle potential NaNs in training data before fitting
        X_pos['bats'].fillna('Unknown', inplace=True)
        X_pos['throws'].fillna('Unknown', inplace=True)
        X_pos['age'].fillna(X_pos['age'].median(), inplace=True)
        X_pos['height_in'].fillna(X_pos['height_in'].median(), inplace=True)
        X_pos['weight_int'].fillna(X_pos['weight_int'].median(), inplace=True)
        y_pos.fillna('Unknown', inplace=True)

        categorical_features = ["bats", "throws"]
        categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))])

        numeric_features_pos = ["age", "height_in", "weight_int"]
        numeric_transformer_pos = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

        preprocess_pos = ColumnTransformer(transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer_pos, numeric_features_pos)
        ])

        rf_pos = Pipeline(steps=[
            ("preprocess", preprocess_pos),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=2025))
        ])
        rf_pos.fit(X_pos, y_pos)

        joblib.dump(rf_pos, self.pos_model_path)
        return rf_pos

    def _load_or_train_models(self):
        os.makedirs("models", exist_ok=True)

        if os.path.exists(self.bat_model_path):
            rf_bat = joblib.load(self.bat_model_path)
        else:
            rf_bat = self._train_batting_model()

        if os.path.exists(self.pos_model_path):
            rf_pos = joblib.load(self.pos_model_path)
        else:
            rf_pos = self._train_fielding_model()

        return rf_bat, rf_pos


    def get_starting_lineup(self):
        team_id_series = self.df_rosters.loc[self.df_rosters["team_name"] == self.team_name, "team_id"]
        if team_id_series.empty:
            return pd.DataFrame({"error": [f"Team '{self.team_name}' not found."]})
        team_id = team_id_series.iloc[0]

        team_roster = self.df_rosters[self.df_rosters["team_id"] == team_id]

        # injured_players = self.df_inj.loc[
        #     self.df_inj["team_id"] == team_id, "player_name"].str.strip().dropna().unique()
        # team_roster = team_roster[~team_roster["player_name"].isin(injured_players)].copy()

        team_df = pd.merge(team_roster, self.df_stats, left_on="player_name", right_on="name", how="left")

        bat_feat_cols = ["on_base_percentage", "slugging_percentage", "batting_avg", "home_runs", "doubles", "triples",
                         "stolen_bases", "strikeouts"]
        field_feat_cols = ["bats", "throws", "age", "height_in", "weight_int"]

        # Fill NaNs for prediction using modern pandas syntax
        for col in bat_feat_cols:
            if col in team_df.columns:
                team_df[col] = team_df[col].fillna(self.df_stats[col].median())
        for col in field_feat_cols:
            if col in team_df.columns and team_df[col].dtype == 'object':
                team_df[col] = team_df[col].fillna('Unknown')
            elif col in team_df.columns:
                team_df[col] = team_df[col].fillna(self.df_rosters[col].median())

        team_df["pred_bat_order"] = self.rf_bat.predict(team_df[bat_feat_cols])
        team_df["pred_position"] = self.rf_pos.predict(team_df[field_feat_cols])


        required_positions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
        lineup_rows = []



        pitchers = team_df[team_df["player_position"].isin(["SP", "RP"])]
        if not pitchers.empty:
            sp_candidates = pitchers[pitchers["player_position"] == "SP"]
            sp = sp_candidates.sort_values("earned_run_avg").iloc[0] if not sp_candidates.empty else \
                pitchers.sort_values("earned_run_avg").iloc[0]
            


            starter_prob = 100 / (1 + sp["earned_run_avg"]) if pd.notna(sp["earned_run_avg"]) else 50
            lineup_rows.append({
                "team_name": self.team_name, "position_group": "Pitchers", "player_name": sp["player_name"],
                "player_number": sp["player_number"], "player_position": sp["player_position"], "bats": sp["bats"],
                "batting_spot": "—", "status": sp["status"], "starter_probability": starter_prob, 
                'player_id': sp['player_id_x'],
            })

        position_players = team_df[~team_df["player_position"].isin(["SP", "RP"])].sort_values("pred_bat_order")
        chosen_ids = set()

        for pos in required_positions:
            candidates = position_players[
                (position_players["pred_position"] == pos) & (~position_players.index.isin(chosen_ids))]
            if candidates.empty:
                candidates = position_players[~position_players.index.isin(chosen_ids)]

            if not candidates.empty:
                player_row = candidates.iloc[0]
                chosen_ids.add(player_row.name)
                starter_prob = 100 - ((player_row["pred_bat_order"] - 1) * 100 / 9)

                lineup_rows.append({
                    "team_name": self.team_name, "position_group": "Fielders", "player_name": player_row["player_name"],
                    "player_number": player_row["player_number"], "player_position": pos, "bats": player_row["bats"],
                    "throws": player_row["throws"], "batting_spot": int(player_row["pred_bat_order"]),
                    "status": player_row["status"], "starter_probability": starter_prob, 
                    'player_id': player_row['player_id_x'],
                })

        remaining = position_players[~position_players.index.isin(chosen_ids)]
        if not remaining.empty:
            dh_player = remaining.iloc[0]
            starter_prob = 100 - ((dh_player["pred_bat_order"] - 1) * 100 / 9)

            lineup_rows.append({
                "team_name": self.team_name, "position_group": "Designated Hitter",
                "player_name": dh_player["player_name"],
                "player_number": dh_player["player_number"], "player_position": "DH", "bats": dh_player["bats"],
                "throws": dh_player["throws"], "batting_spot": int(dh_player["pred_bat_order"]),
                "status": dh_player["status"], "starter_probability": starter_prob,
                'player_id': dh_player['player_id_x'],
            })

        df_lineup = pd.DataFrame(lineup_rows)
        # Add position numbers for the new schematic view
        pos_map = {"C": 2, "P": 1, "1B": 3, "2B": 4, "SS": 6, "3B": 5, "LF": 7, "CF": 8, "RF": 9, "DH": 0}
        df_lineup['position_number'] = df_lineup['player_position'].map(pos_map)

        df_lineup = df_lineup.replace({np.nan: None})
        df_lineup['player_photo'] = df_lineup['player_id'].apply(lambda pid: f"{IMAGE_URL[:-1]}-mlb/{pid}.png")
        return df_lineup


