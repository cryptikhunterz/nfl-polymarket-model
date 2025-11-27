"""
Pro Football Focus (PFF) Integration Module

Provides access to PFF metrics for:
- Line Play (20% of model): OL Pass Block Win Rate, Pressure Rate
- Luck Regression (10% of model): Interceptable Passes, QB Grade

Supports multiple data sources:
1. PFF API (requires subscription)
2. CSV uploads (manual data entry)
3. Mock data (for testing)
"""

import os
import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


class PFFDataSource(Enum):
    """Available PFF data sources"""
    API = "api"          # Direct PFF API (requires credentials)
    CSV = "csv"          # CSV file uploads
    MOCK = "mock"        # Mock data for testing


@dataclass
class PFFConfig:
    """PFF Integration Configuration"""
    data_source: PFFDataSource = PFFDataSource.MOCK
    api_key: Optional[str] = None
    csv_path: Optional[str] = None
    cache_hours: int = 24

    @classmethod
    def load(cls, config_path: str = None) -> 'PFFConfig':
        """Load config from file or environment"""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                data = json.load(f)
                return cls(
                    data_source=PFFDataSource(data.get('data_source', 'mock')),
                    api_key=data.get('api_key'),
                    csv_path=data.get('csv_path'),
                    cache_hours=data.get('cache_hours', 24)
                )

        # Try environment variables
        api_key = os.environ.get('PFF_API_KEY')
        if api_key:
            return cls(data_source=PFFDataSource.API, api_key=api_key)

        return cls()  # Default to mock

    def save(self, config_path: str):
        """Save config to file"""
        data = {
            'data_source': self.data_source.value,
            'api_key': self.api_key,
            'csv_path': self.csv_path,
            'cache_hours': self.cache_hours
        }
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)


class PFFIntegration:
    """
    Pro Football Focus data integration

    Provides PFF metrics for:
    - Line Play Factor (20%): OL performance metrics
    - Luck Regression Factor (10%): Interceptable passes, QB grade
    """

    # NFL team abbreviations
    NFL_TEAMS = [
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
    ]

    def __init__(self, config: PFFConfig = None):
        """Initialize PFF integration"""
        self.config = config or PFFConfig.load()
        self._cache = {}
        self._cache_time = None

        # Data directory
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def get_line_play_metrics(self, season: int = 2024) -> pd.DataFrame:
        """
        Get offensive line metrics for Line Play factor

        Returns DataFrame with columns:
        - team: Team abbreviation
        - ol_pass_block_win_rate: OL Pass Block Win Rate (0-100)
        - ol_run_block_win_rate: OL Run Block Win Rate (0-100)
        - pressure_rate_allowed: Pressure Rate Allowed (0-100, lower is better)
        - sacks_allowed_per_game: Sacks allowed per game
        - line_play_score: Composite score (0-100)
        """
        cache_key = f"line_play_{season}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Always use CSV data - no mock data fallback
        df = self._load_line_play_csv(season)

        self._cache[cache_key] = df
        self._cache_time = datetime.now()

        return df

    def get_luck_regression_metrics(self, season: int = 2024) -> pd.DataFrame:
        """
        Get luck regression metrics

        Returns DataFrame with columns:
        - team: Team abbreviation
        - interceptable_passes: Interceptable passes thrown
        - actual_interceptions: Actual interceptions thrown
        - int_luck_score: INT luck score (positive = unlucky, negative = lucky)
        - qb_grade: PFF QB Grade (0-100)
        - turnover_luck_score: Overall turnover luck (-100 to 100)
        """
        cache_key = f"luck_{season}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Always use CSV data - no mock data fallback
        df = self._load_luck_csv(season)

        self._cache[cache_key] = df
        self._cache_time = datetime.now()

        return df

    def get_all_metrics(self, season: int = 2024) -> pd.DataFrame:
        """Get all PFF metrics combined"""
        line_play = self.get_line_play_metrics(season)
        luck = self.get_luck_regression_metrics(season)

        # Merge on team
        df = line_play.merge(luck, on='team', how='outer')
        return df

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is still valid"""
        if key not in self._cache or self._cache_time is None:
            return False

        hours_elapsed = (datetime.now() - self._cache_time).total_seconds() / 3600
        return hours_elapsed < self.config.cache_hours

    def _fetch_line_play_api(self, season: int) -> pd.DataFrame:
        """Fetch line play metrics from PFF API"""
        # Placeholder for actual API implementation
        # Would require PFF API subscription
        raise NotImplementedError(
            "PFF API integration requires subscription. "
            "Use CSV upload or mock data instead."
        )

    def _fetch_luck_api(self, season: int) -> pd.DataFrame:
        """Fetch luck metrics from PFF API"""
        raise NotImplementedError(
            "PFF API integration requires subscription. "
            "Use CSV upload or mock data instead."
        )

    def _load_line_play_csv(self, season: int) -> pd.DataFrame:
        """Load line play metrics from CSV"""
        # First try the PFF team grades template (scraped from PFF)
        stats_dir = Path(__file__).parent.parent / "data" / "stats"
        pff_grades_path = stats_dir / "pff_team_grades_template.csv"

        if pff_grades_path.exists():
            df = pd.read_csv(pff_grades_path)
            # Map PFF grades to line play metrics
            if 'pblk' in df.columns and 'team' in df.columns:
                result = pd.DataFrame({
                    'team': df['team'],
                    'ol_pass_block_win_rate': df['pblk'],  # PFF Pass Block Grade
                    'ol_run_block_win_rate': df['rblk'] if 'rblk' in df.columns else df['pblk'] * 0.9,
                    'pressure_rate_allowed': 100 - df['pblk'],  # Inverse of pass block
                    'sacks_allowed_per_game': (100 - df['pblk']) / 25,  # Estimate
                })
                result['line_play_score'] = (
                    result['ol_pass_block_win_rate'] * 0.5 +
                    (100 - result['pressure_rate_allowed']) * 0.3 +
                    result['ol_run_block_win_rate'] * 0.2
                )
                print(f"Loaded PFF line play data from {pff_grades_path}")
                return result

        # If no PFF grades template found, raise error - no mock data fallback
        raise ValueError(
            "PFF team grades CSV not found. Please upload pff_team_grades_template.csv to data/stats/"
        )

    def _load_luck_csv(self, season: int) -> pd.DataFrame:
        """Load luck metrics from CSV - uses REAL PFF QB passing grades data"""
        stats_dir = Path(__file__).parent.parent / "data" / "stats"

        # First try the QB passing grades CSV (has real TWP data)
        qb_grades_path = stats_dir / "pff_qb_passing_grades.csv"
        team_grades_path = stats_dir / "pff_team_grades_template.csv"

        if qb_grades_path.exists():
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Length of header or names does not match length of data')
                qb_df = pd.read_csv(qb_grades_path, index_col=False)

            # Convert numeric columns
            numeric_cols = ['games', 'twp', 'int', 'pass_grade', 'att', 'com', 'td']
            for col in numeric_cols:
                if col in qb_df.columns:
                    qb_df[col] = pd.to_numeric(qb_df[col], errors='coerce').fillna(0)

            # Map QB data to team-level luck metrics
            # Group by team and get primary QB (most games played)
            if 'team' in qb_df.columns and 'twp' in qb_df.columns:
                # Get primary QB per team (most games played)
                def get_primary_qb(x):
                    if 'games' in x.columns and len(x) > 0:
                        return x.loc[x['games'].idxmax()]
                    return x.iloc[0]

                team_qbs = qb_df.groupby('team').apply(get_primary_qb).reset_index(drop=True)

                result_data = []
                for _, row in team_qbs.iterrows():
                    team = row['team']
                    twp = row['twp'] if pd.notna(row.get('twp')) else 15  # Turnover Worthy Plays (real data)
                    # Column is 'int' not 'actual_int'
                    actual_int = row['int'] if 'int' in row and pd.notna(row.get('int')) else 10
                    pass_grade = row['pass_grade'] if 'pass_grade' in row and pd.notna(row.get('pass_grade')) else 65

                    # Calculate luck score: TWP = interceptable passes proxy
                    # If actual INTs < TWP, QB is "lucky" (throws should have been picked)
                    # If actual INTs > TWP, QB is "unlucky" (getting picked on non-risky throws)
                    int_luck_score = actual_int - twp

                    result_data.append({
                        'team': team,
                        'interceptable_passes': int(twp),  # TWP = turnover worthy plays
                        'actual_interceptions': int(actual_int),
                        'int_luck_score': round(int_luck_score, 1),
                        'qb_grade': round(pass_grade, 1),
                        'turnover_luck_score': round(int_luck_score * 3, 1)
                    })

                print(f"Loaded REAL PFF QB luck data from {qb_grades_path} ({len(result_data)} teams)")
                return pd.DataFrame(result_data)

        # Fallback to team grades template with pass grade
        if team_grades_path.exists():
            df = pd.read_csv(team_grades_path)
            if 'pass' in df.columns and 'team' in df.columns:
                result_data = []
                for _, row in df.iterrows():
                    team = row['team']
                    pass_grade = row.get('pass', 70)

                    # Estimate TWP from pass grade (higher grade = fewer TWP)
                    # This is an estimate until real QB data is uploaded
                    twp_estimate = int(25 - (pass_grade - 60) * 0.3)
                    twp_estimate = max(5, min(25, twp_estimate))

                    # Estimate actual INTs (similar to TWP for neutral luck)
                    actual_int_estimate = twp_estimate

                    result_data.append({
                        'team': team,
                        'interceptable_passes': twp_estimate,
                        'actual_interceptions': actual_int_estimate,
                        'int_luck_score': 0,  # Neutral until real data
                        'qb_grade': round(pass_grade, 1),
                        'turnover_luck_score': 0
                    })

                print(f"Loaded PFF luck data from team grades {team_grades_path} (estimated TWP)")
                return pd.DataFrame(result_data)

        # If no CSV data found, raise error - no mock data fallback
        raise ValueError(
            "PFF QB passing grades CSV not found. Please upload pff_qb_passing_grades.csv or pff_team_grades_template.csv to data/stats/"
        )

    def _generate_mock_line_play(self, season: int) -> pd.DataFrame:
        """Generate realistic mock line play data based on 2025 season performance"""
        import random
        random.seed(42 + season)  # Reproducible

        # 2025 season team tendencies based on actual standings and performance
        # Updated with current 2025 records: DET 10-2, KC 9-2, PHI 9-2, etc.
        team_tendencies = {
            # Elite OLs (Top tier teams)
            'DET': {'ol_boost': 12},   # 10-2, elite OL
            'PHI': {'ol_boost': 11},   # 9-2, historically elite OL
            'KC': {'ol_boost': 9},     # 9-2, protect Mahomes well
            'BAL': {'ol_boost': 8},    # 8-4, strong run blocking
            'BUF': {'ol_boost': 8},    # 8-3, improved OL
            'PIT': {'ol_boost': 7},    # 8-3, improved protection
            'MIN': {'ol_boost': 6},    # 8-3, surprise good season
            'GB': {'ol_boost': 5},     # 7-4, solid OL
            'SF': {'ol_boost': 5},     # 6-5, still good despite injuries
            'LAC': {'ol_boost': 4},    # 7-4, improved with Harbaugh
            'HOU': {'ol_boost': 4},    # 6-5, protect CJ Stroud
            'DEN': {'ol_boost': 4},    # 9-2, Sean Payton coaching
            'IND': {'ol_boost': 3},    # 8-3, solid OL
            'SEA': {'ol_boost': 3},    # 6-5
            'WAS': {'ol_boost': 2},    # 8-3, protect Daniels
            'TB': {'ol_boost': 2},     # 8-4
            'ATL': {'ol_boost': 1},    # 4-7
            'LAR': {'ol_boost': 1},    # 4-7
            'DAL': {'ol_boost': 0},    # 5-5-1, OL struggles
            'NO': {'ol_boost': 0},     # 2-9
            'CIN': {'ol_boost': -1},   # 3-8, OL issues
            'CLE': {'ol_boost': -2},   # 3-8
            'JAX': {'ol_boost': -3},   # 7-4
            'MIA': {'ol_boost': -3},   # 4-7, protection issues
            'ARI': {'ol_boost': -4},   # 3-8
            'NYJ': {'ol_boost': -5},   # 2-9, terrible OL
            'CHI': {'ol_boost': -5},   # 6-6, Caleb Williams running for his life
            'TEN': {'ol_boost': -5},   # 1-10
            'LV': {'ol_boost': -6},    # 2-9
            'NE': {'ol_boost': -6},    # 2-9
            'NYG': {'ol_boost': -7},   # 2-10
            'CAR': {'ol_boost': -8},   # 6-5, bad OL but Bryce Young playing better
        }

        data = []
        for team in self.NFL_TEAMS:
            boost = team_tendencies.get(team, {}).get('ol_boost', 0)

            # Base metrics with team tendency
            ol_pass_block = 55 + boost + random.gauss(0, 5)
            ol_run_block = 52 + boost * 0.8 + random.gauss(0, 6)
            pressure_rate = 30 - boost * 0.5 + random.gauss(0, 4)
            sacks_per_game = 2.5 - boost * 0.08 + random.gauss(0, 0.5)

            # Clamp values
            ol_pass_block = max(35, min(75, ol_pass_block))
            ol_run_block = max(35, min(75, ol_run_block))
            pressure_rate = max(15, min(45, pressure_rate))
            sacks_per_game = max(1.0, min(4.5, sacks_per_game))

            # Composite score
            line_play_score = (
                ol_pass_block * 0.4 +
                ol_run_block * 0.3 +
                (100 - pressure_rate * 2) * 0.3
            )

            data.append({
                'team': team,
                'ol_pass_block_win_rate': round(ol_pass_block, 1),
                'ol_run_block_win_rate': round(ol_run_block, 1),
                'pressure_rate_allowed': round(pressure_rate, 1),
                'sacks_allowed_per_game': round(sacks_per_game, 2),
                'line_play_score': round(line_play_score, 1)
            })

        return pd.DataFrame(data)

    def _generate_mock_luck(self, season: int) -> pd.DataFrame:
        """Generate realistic mock luck regression data based on 2025 season"""
        import random
        random.seed(43 + season)  # Different seed

        # 2025 season luck tendencies based on turnover differential and close games
        luck_tendencies = {
            # "Lucky" teams - fewer INTs than expected, win close games
            'DET': {'luck_boost': 6, 'qb_boost': 15},    # 10-2, Goff playing well
            'KC': {'luck_boost': 5, 'qb_boost': 12},     # 9-2, Mahomes magic
            'DEN': {'luck_boost': 5, 'qb_boost': 5},     # 9-2, overperforming
            'PHI': {'luck_boost': 4, 'qb_boost': 10},    # 9-2, Hurts playing well
            'MIN': {'luck_boost': 4, 'qb_boost': 12},    # 8-3, Sam Darnold resurgence
            'PIT': {'luck_boost': 3, 'qb_boost': 5},     # 8-3, close game wins
            'WAS': {'luck_boost': 3, 'qb_boost': 8},     # 8-3, Jayden Daniels ROY candidate
            'IND': {'luck_boost': 2, 'qb_boost': 3},     # 8-3
            'BUF': {'luck_boost': 2, 'qb_boost': 10},    # 8-3, Josh Allen MVP candidate
            'TB': {'luck_boost': 2, 'qb_boost': 5},      # 8-4, Baker mayfield good
            'BAL': {'luck_boost': 1, 'qb_boost': 15},    # 8-4, Lamar MVP candidate
            'GB': {'luck_boost': 1, 'qb_boost': 6},      # 7-4, Jordan Love solid
            'LAC': {'luck_boost': 1, 'qb_boost': 5},     # 7-4, Herbert good
            'JAX': {'luck_boost': 0, 'qb_boost': 2},     # 7-4, Lawrence inconsistent
            'SF': {'luck_boost': 0, 'qb_boost': 5},      # 6-5, injuries
            'SEA': {'luck_boost': 0, 'qb_boost': 4},     # 6-5, Geno Smith okay
            'HOU': {'luck_boost': 0, 'qb_boost': 8},     # 6-5, Stroud good
            'CAR': {'luck_boost': -1, 'qb_boost': -5},   # 6-5, Bryce Young improved
            'CHI': {'luck_boost': -1, 'qb_boost': 5},    # 6-6, Caleb Williams promising
            'DAL': {'luck_boost': -2, 'qb_boost': 2},    # 5-5-1, Dak out
            'ATL': {'luck_boost': -2, 'qb_boost': -2},   # 4-7, Kirk Cousins struggling
            'MIA': {'luck_boost': -2, 'qb_boost': -5},   # 4-7, Tua injuries
            'LAR': {'luck_boost': -2, 'qb_boost': 3},    # 4-7, Stafford aging
            # "Unlucky" teams - more INTs than expected, lose close games
            'CIN': {'luck_boost': -3, 'qb_boost': 8},    # 3-8, Burrow good but unlucky
            'ARI': {'luck_boost': -3, 'qb_boost': 0},    # 3-8, Murray inconsistent
            'CLE': {'luck_boost': -4, 'qb_boost': -8},   # 3-8, QB carousel
            'LV': {'luck_boost': -4, 'qb_boost': -5},    # 2-9, Gardner Minshew
            'NO': {'luck_boost': -4, 'qb_boost': -5},    # 2-9, Derek Carr struggling
            'NE': {'luck_boost': -5, 'qb_boost': -3},    # 2-9, Drake Maye learning
            'NYG': {'luck_boost': -5, 'qb_boost': -10},  # 2-10, Daniel Jones cut
            'NYJ': {'luck_boost': -6, 'qb_boost': 0},    # 2-9, Aaron Rodgers decline
            'TEN': {'luck_boost': -6, 'qb_boost': -5},   # 1-10, worst team
        }

        data = []
        for team in self.NFL_TEAMS:
            team_data = luck_tendencies.get(team, {})
            luck_boost = team_data.get('luck_boost', 0)
            qb_boost = team_data.get('qb_boost', 0)

            # Interceptable passes (season total, ~15-30 range)
            interceptable = 20 + random.gauss(0, 4)

            # Actual INTs (varies based on luck)
            # Lucky teams: fewer actual INTs than interceptable
            # Unlucky teams: more actual INTs than interceptable
            actual_int = interceptable - luck_boost * 0.3 + random.gauss(0, 2)
            actual_int = max(5, actual_int)  # Minimum 5 INTs

            # INT luck score (positive = unlucky, got more INTs than expected)
            int_luck_score = actual_int - interceptable

            # QB Grade (0-100) - use team-specific QB boost
            qb_grade = 65 + qb_boost + random.gauss(0, 5)
            qb_grade = max(40, min(95, qb_grade))

            # Overall turnover luck (negative = lucky, positive = unlucky)
            turnover_luck = int_luck_score * 3 + random.gauss(0, 3)
            turnover_luck = max(-30, min(30, turnover_luck))

            data.append({
                'team': team,
                'interceptable_passes': round(interceptable, 0),
                'actual_interceptions': round(actual_int, 0),
                'int_luck_score': round(int_luck_score, 1),
                'qb_grade': round(qb_grade, 1),
                'turnover_luck_score': round(turnover_luck, 1)
            })

        return pd.DataFrame(data)

    def upload_csv(self, file_path: str, metric_type: str) -> bool:
        """
        Upload a CSV file for PFF metrics

        Args:
            file_path: Path to the CSV file
            metric_type: 'line_play' or 'luck'

        Returns:
            True if successful
        """
        if metric_type not in ['line_play', 'luck']:
            raise ValueError("metric_type must be 'line_play' or 'luck'")

        df = pd.read_csv(file_path)

        # Validate
        if metric_type == 'line_play':
            required = ['team', 'ol_pass_block_win_rate', 'pressure_rate_allowed']
        else:
            required = ['team', 'interceptable_passes', 'actual_interceptions']

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Validate team names
        invalid_teams = set(df['team']) - set(self.NFL_TEAMS)
        if invalid_teams:
            raise ValueError(f"Invalid team abbreviations: {invalid_teams}")

        # Save to data directory
        output_path = self.data_dir / f"pff_{metric_type}_2024.csv"
        df.to_csv(output_path, index=False)

        # Clear cache
        self._cache = {}

        print(f"Saved PFF {metric_type} data to {output_path}")
        return True

    def get_data_status(self) -> Dict[str, Any]:
        """Get status of PFF data"""
        line_play_csv = self.data_dir / "pff_line_play_2024.csv"
        luck_csv = self.data_dir / "pff_luck_2024.csv"

        return {
            'data_source': self.config.data_source.value,
            'has_line_play_csv': line_play_csv.exists(),
            'has_luck_csv': luck_csv.exists(),
            'cache_hours': self.config.cache_hours,
            'api_configured': self.config.api_key is not None,
            'line_play_csv_path': str(line_play_csv) if line_play_csv.exists() else None,
            'luck_csv_path': str(luck_csv) if luck_csv.exists() else None,
        }


def get_pff_integration() -> PFFIntegration:
    """Get singleton PFF integration instance"""
    config_path = Path(__file__).parent.parent / "config" / "pff_config.json"
    config = PFFConfig.load(str(config_path))
    return PFFIntegration(config)


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PFF Integration')
    parser.add_argument('command', choices=['status', 'line_play', 'luck', 'all'],
                       help='Command to run')
    parser.add_argument('--season', type=int, default=2024, help='Season year')

    args = parser.parse_args()

    pff = get_pff_integration()

    if args.command == 'status':
        status = pff.get_data_status()
        print("\nPFF Data Status:")
        print("-" * 40)
        for k, v in status.items():
            print(f"  {k}: {v}")

    elif args.command == 'line_play':
        df = pff.get_line_play_metrics(args.season)
        print(f"\nLine Play Metrics ({args.season}):")
        print("-" * 60)
        print(df.to_string(index=False))

    elif args.command == 'luck':
        df = pff.get_luck_regression_metrics(args.season)
        print(f"\nLuck Regression Metrics ({args.season}):")
        print("-" * 60)
        print(df.to_string(index=False))

    elif args.command == 'all':
        df = pff.get_all_metrics(args.season)
        print(f"\nAll PFF Metrics ({args.season}):")
        print("-" * 80)
        print(df.to_string(index=False))
