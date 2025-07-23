import pandas as pd
import logging
from app.config import GOALSERVE_BASE_URL, GOALSERVE_API_KEY
from app.services.NFL.data_processor import NflDataProcessor
from app.services.helper import safe_float, safe_int

class NflPlayerDataProcessor:
    def __init__(self):
        self.api_key = GOALSERVE_API_KEY
        self.base_url = GOALSERVE_BASE_URL
        self.processor = NflDataProcessor()

    tag_fields = {
        "passing": ["comp_att", "yards", "average", "passing_touch_downs", "interceptions", "sacks", "rating", "two_pt"],
        "rushing": ["total_rushes", "yards", "average", "rushing_touch_downs", "longest_rush", "two_pt"],
        "receiving": ["targets", "total_receptions", "yards", "average", "receiving_touch_downs", "longest_reception", "two_pt"],
        "fumbles": ["total", "lost", "rec", "rec_td"],
        "interceptions": ["total_interceptions", "yards", "intercepted_touch_downs"],
        "defensive": ["tackles", "unassisted_tackles", "sacks", "tfl", "passes_defended", "qb_hts", "interceptions_for_touch_downs", "blocked_kicks", "kick_return_td", "exp_return_td", "ff"],
        "kick_returns": ["total", "yards", "average", "lg", "kick_return_td", "exp_return_td"],
        "punt_returns": ["total", "yards", "average", "lg", "td", "kick_return_td", "exp_return_td"],
        "kicking": ["field_goals", "pct", "long", "extra_point", "points", "field_goals_from_1_19_yards", "field_goals_from_20_29_yards", "field_goals_from_30_39_yards", "field_goals_from_40_49_yards", "field_goals_from_50_yards"],
        "punting": ["total", "yards", "average", "touchbacks", "in20", "lg"],
        "scoring": ["total_touch_downs", "extra_points", "field_goals", "safeties", "points"]
    }

    def fetch_player_stats(self, team_id: str) -> pd.DataFrame:
        endpoint = f"football/{team_id}_player_stats"
        stats_data = self.processor.fetch_data(endpoint)

        if not stats_data or 'statistic' not in stats_data:
            logging.warning(f"No stats data found for team {team_id}")
            return pd.DataFrame()

        stat_root = stats_data['statistic']
        team_id_val = stat_root.get('@id', team_id)
        team_name = stat_root.get('@team', '')

        categories = stat_root.get('category', [])
        if isinstance(categories, dict):
            categories = [categories]

        all_records = []
        for cat in categories:
            category_name = cat.get('@name', '') or cat.get('name', '')
            players = cat.get('player', [])
            if isinstance(players, dict):
                players = [players]
            for player in players:
                player_id = player.get('@id') or player.get('@player_id')
                if not player_id:
                    continue
                rec = {
                    "team_id": team_id_val,
                    "team_name": team_name,
                    "player_id": player_id,
                    "player_name": player.get('@name', '') or player.get('name', ''),
                    "position": player.get('@position', ''),
                    "category": category_name,
                }
                for k, v in player.items():
                    if k in ['@id', '@name', '@position', 'name', 'position']:
                        continue
                    key = k.lstrip('@')
                    if isinstance(v, str) and v.replace(',', '').replace('.', '').isdigit():
                        v_clean = v.replace(',', '')
                        rec[key] = safe_float(v_clean) if '.' in v_clean else safe_int(v_clean)
                    else:
                        rec[key] = v
                all_records.append(rec)
        return pd.DataFrame(all_records)

    def fetch_roster(self, team_id: str) -> pd.DataFrame:
        endpoint = f"football/{team_id}_rosters"
        roster_data = self.processor.fetch_data(endpoint)
        team = roster_data.get('team', {})
        positions = team.get('position', [])
        if isinstance(positions, dict):
            positions = [positions]

        records = []
        for pos in positions:
            players = pos.get('player', [])
            if isinstance(players, dict):
                players = [players]
            for p in players:
                records.append({
                    "team_id": team.get("@id"),
                    "team_name": team.get("@name"),
                    "player_id": p.get("@id", "") or p.get("@player_id", ""),
                    "player_name": p.get("@name"),
                    "position": p.get("@position", ""),
                    "jersey_number": p.get("@number", ""),
                })
        return pd.DataFrame(records)

    def parse_injuries_to_df(self, injuries_data: dict) -> pd.DataFrame:
        injuries = []
        teams = injuries_data.get('injuries', {}).get('team', [])
        if isinstance(teams, dict):
            teams = [teams]

        for team in teams:
            reports = team.get('report', [])
            if isinstance(reports, dict):
                reports = [reports]
            for r in reports:
                injuries.append({
                    'team_name': team.get('@name', ''),
                    'team_id': team.get('@id', ''),
                    'player_id': r.get('@player_id', ''),
                    'player_name': r.get('@player_name', ''),
                    'date': r.get('@date', ''),
                    'status': r.get('@status', ''),
                    'description': r.get('@description', ''),
                })
        return pd.DataFrame(injuries)

    def score_row(self, row: pd.Series) -> float:
        pos = row.get('position', '').upper()

        if pos == 'QB':
            return round(row.get("passing_yards", 0) * 0.04 + row.get("passing_touch_downs", 0) * 4 - row.get("interceptions", 0) * 2 + row.get("rushing_yards", 0) * 0.1 + row.get("rushing_touch_downs", 0) * 6, 2)
        elif pos == 'RB':
            return round(row.get("rushing_yards", 0) * 0.1 + row.get("rushing_touch_downs", 0) * 6 + row.get("receiving_yards", 0) * 0.1 + row.get("receiving_touch_downs", 0) * 6 + row.get("total_receptions", 0), 2)
        elif pos in ['WR', 'TE']:
            return round(row.get("receiving_yards", 0) * 0.1 + row.get("receiving_touch_downs", 0) * 6 + row.get("total_receptions", 0), 2)
        elif pos in ['K', 'PK']:
            return round(row.get("field_goals", 0) * 3 + (2 if row.get("long", 0) >= 50 else 0) + row.get("extra_point", 0), 2)
        return 0

    def build_player_feature_set(self, team_id: str, injuries_data: dict) -> pd.DataFrame:
        roster = self.fetch_roster(team_id)
        stats = self.fetch_player_stats(team_id)
        injuries = self.parse_injuries_to_df(injuries_data)

        stats_pivot = stats.pivot_table(index="player_id", columns="category", aggfunc='first').reset_index()
        stats_pivot.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stats_pivot.columns]

        merged = pd.merge(roster, stats_pivot, on="player_id", how="left")
        merged = pd.merge(merged, injuries, on="player_id", how="left")

        merged['fantasy_score'] = merged.apply(self.score_row, axis=1)
        return merged.sort_values(by='fantasy_score', ascending=False).fillna(0)
