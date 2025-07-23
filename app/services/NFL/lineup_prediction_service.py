import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns



class NFLLineupPredictor:
    """Predict starting line‑ups for an NFL team and persist the model as a .pkl file."""

    # --------------------------- constants ----------------------------- #

    POSITION_MAPPING = {
        # Offense     1‑5
        "QB": 1, "RB": 2, "FB": 2, "WR": 3, "TE": 4,
        "C": 5, "G": 5, "OT": 5, "OL": 5,
        # Defense     6‑10
        "DE": 6, "DT": 7, "LB": 8, "CB": 9, "S": 10,
        # Specialists 11‑13
        "K": 11, "P": 12, "LS": 13,
    }

    UNIT_MAP = {
        # offense
        "QB": "offense", "RB": "offense", "FB": "offense", "WR": "offense", "TE": "offense",
        "C": "offense", "G": "offense", "OT": "offense", "OL": "offense",
        # defense
        "DE": "defense", "DT": "defense", "LB": "defense", "CB": "defense", "S": "defense",
        # specials
        "K": "special", "P": "special", "LS": "special",
    }

    DEFAULT_STARTER_COUNTS = {
        # offense (11)
        "QB": 1, "RB": 1, "WR": 3, "TE": 1, "C": 1, "G": 2, "OT": 2,
        # defense (11)
        "DE": 2, "DT": 2, "LB": 3, "CB": 2, "S": 2,
        # specialists (3)
        "K": 1, "P": 1, "LS": 1,
    }

    # ------------------------------------------------------------------ #

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns: list[str] | None = None
        self.full_data: pd.DataFrame | None = None

    # ============================= helpers ============================ #

    def separate_by_unit(self, df: pd.DataFrame):
        """Return three dataframes (offense, defense, special) with unique player rows."""
        df_unique = df.drop_duplicates(subset="player_name", keep="first")
        return (
            df_unique[df_unique.unit_group == "offense"],
            df_unique[df_unique.unit_group == "defense"],
            df_unique[df_unique.unit_group == "special"],
        )

    @staticmethod
    def _clean_team_ids(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = [c for c in df.columns if c.startswith("team_id")]
        if len(cols) == 2:
            keep = "team_id_y" if "team_id_y" in cols else cols[0]
            df = df.drop(columns=[c for c in cols if c != keep]).rename(columns={keep: "team_id"})
        elif len(cols) == 1 and cols[0] != "team_id":
            df = df.rename(columns={cols[0]: "team_id"})
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
        return df

    # ============================ ETL steps =========================== #

    def load_and_preprocess_data(self, info_csv: str, stats_csv: str) -> pd.DataFrame:
        info = pd.read_csv(info_csv)
        stats = pd.read_csv(stats_csv)
        merged = info.merge(stats, left_on="player_id", right_on="id", how="inner", suffixes=("_info", ""))
        self.full_data = self._clean_team_ids(merged).reset_index(drop=True)
        return self.full_data

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()
        num_cols = [
            "completion_pct", "passing_attempts", "passing_touchdowns", "yards", "yards_per_rush_avg",
            "rushing_attempts", "rushing_touchdowns", "receiving_targets", "receptions", "receiving_yards",
            "receiving_touchdowns", "total_tackles", "sacks", "interceptions", "forced_fumbles",
            "total_points",
        ]
        f[num_cols] = f[num_cols].apply(pd.to_numeric, errors="coerce")
        f["position_group"] = f.player_position.map(self.POSITION_MAPPING)
        f["unit_group"] = f.player_position.map(self.UNIT_MAP).fillna("unknown")
        f["player_number_filled"] = f.player_number.fillna(0)
        for col in num_cols:
            f[f"{col}_filled"] = f[col].fillna(0)

        # derived features
        f["passing_efficiency"] = f.completion_pct.fillna(0) * f.passing_attempts.fillna(0) / 100
        f["rushing_efficiency"] = f.rushing_attempts.fillna(0) * f.yards_per_rush_avg.fillna(0)
        f["receiving_efficiency"] = f.receptions.fillna(0) * f.receiving_yards.fillna(0) / 100
        f["defensive_impact"] = f.total_tackles.fillna(0) + 2 * f.sacks.fillna(0) + 3 * f.interceptions.fillna(0)
        return f

    def create_target(self, feats: pd.DataFrame) -> np.ndarray:
        y = np.zeros(len(feats))
        for grp in feats.position_group.dropna().unique():
            m = feats.position_group == grp
            sub = feats.loc[m]
            if grp == 1:  # QB
                y[m] = (sub.passing_attempts >= sub.passing_attempts.quantile(0.7)).astype(int)
            elif grp in (2, 3):  # RB / WR
                y[m] = (sub.yards >= sub.yards.quantile(0.6)).astype(int)
            elif grp in (6, 7, 8):  # DL / LB
                y[m] = (sub.total_tackles >= sub.total_tackles.quantile(0.6)).astype(int)
            else:
                y[m] = (sub.total_points >= sub.total_points.quantile(0.5)).astype(int)
        return y

    def _encode_scale(self, feats: pd.DataFrame) -> pd.DataFrame:
        feats = feats.copy()
        num = [
            "position_group", "completion_pct_filled", "passing_attempts_filled",
            "passing_touchdowns_filled", "yards_filled", "rushing_attempts_filled", "rushing_touchdowns_filled",
            "receiving_targets_filled", "receptions_filled", "receiving_touchdowns_filled", "total_tackles_filled",
            "sacks_filled", "interceptions_filled", "passing_efficiency", "rushing_efficiency",
            "receiving_efficiency", "defensive_impact",
        ]

        # categorical encoding
        for cat in ("team_name", "player_position"):
            le = self.label_encoders.setdefault(cat, LabelEncoder().fit(feats[cat].astype(str)))
            feats[f"{cat}_enc"] = le.transform(feats[cat].astype(str))
            num.append(f"{cat}_enc")

        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(feats[num]), columns=num)
        self.feature_columns = num
        return pd.DataFrame(self.scaler.fit_transform(X), columns=num)

    # ============================== API ============================== #

    def train(self, info_csv: str, stats_csv: str, save_path: str | Path = "nfl_lineup_model.pkl"):
        """Train the model on two CSVs and save it to *save_path* (.pkl)."""
        data = self.load_and_preprocess_data(info_csv, stats_csv)
        feats = self.create_features(data)
        y = self.create_target(feats)
        X = self._encode_scale(feats)
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        self.model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_tr, y_tr)

        # 👉 persist to disk
        self.save(save_path)

    # -------------------------- persistence -------------------------- #

    def save(self, path: str | Path = "nfl_lineup_model.pkl"):
        """Serialize model + transformers to a single .pkl file."""
        payload = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str | Path = "nfl_lineup_model.pkl"):
        """Load a model previously saved with *save()*."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.label_encoders = payload["label_encoders"]
        self.scaler = payload["scaler"]
        self.feature_columns = payload["feature_columns"]

    # ---------------------------------------------------------------- #

    def _team_slice(self, team_id: int) -> pd.DataFrame:
        if self.full_data is None:
            raise RuntimeError("Run train() or load() first")
        return self.full_data[self.full_data.team_id == int(team_id)].copy()

    def predict_lineup(self, team_id: int, prob_threshold: float = 0.5) -> pd.DataFrame:
        """Return probabilities & starter flags for every player on one team."""
        df_raw = self._team_slice(team_id)
        if df_raw.empty:
            return pd.DataFrame()

        feats = self.create_features(df_raw)
        
        X = self._encode_scale(feats)
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= prob_threshold).astype(int)
    
        out = feats[["player_name", "player_number","player_id", "player_position", "team_name", "unit_group"]].copy()
        out["starter_probability"] = probs
        # out["starter_prediction"] = preds
        out["starter_prediction"] = preds
        return out.sort_values(["player_position", "starter_probability"], ascending=[True, False]).reset_index(drop=True)

    def select_starting_lineup(self, df_probs: pd.DataFrame, starter_counts: dict[str, int] | None = None):
        sc = starter_counts or self.DEFAULT_STARTER_COUNTS
        picks = []
        for pos, grp in df_probs.groupby("player_position"):
            n = sc.get(pos, 0)
            if n:
                picks.append(grp.nlargest(n, "starter_probability"))
        if not picks:
            return pd.DataFrame()
        return (
            pd.concat(picks)
            .sort_values(["player_position", "starter_probability"], ascending=[True, False])
            .reset_index(drop=True)
        )

    # ========================== diagnostics ========================= #

    def plot_feature_importance(self):
        if not self.model:
            raise RuntimeError("Model is not trained or loaded.")
        s = (
            pd.Series(self.model.feature_importances_, index=self.feature_columns)
            .sort_values(ascending=False)[:15]
        )
        plt.figure(figsize=(7, 6))
        sns.barplot(x=s.values, y=s.index, orient="h")
        plt.tight_layout()
        plt.show()