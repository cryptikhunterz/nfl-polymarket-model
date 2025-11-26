"""
NFL Factor Calculator Module
Calculates all factors per team for the model:
- QB Quality (30%)
- Team Efficiency (25%)
- Line Play (20%)
- Situational (15%)
- Luck Regression (10%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


class FactorCalculator:
    """Calculate all factors for NFL teams"""

    def __init__(self, current_season: int = 2024, current_week: int = None):
        self.current_season = current_season
        self.current_week = current_week
        self.pbp = None
        self.schedules = None
        self.team_epa = None
        self.qb_stats = None
        self.elos = None

    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")

        # Play-by-play for current and last season
        pbp_files = list(RAW_DIR.glob("pbp_*.csv"))
        if pbp_files:
            dfs = [pd.read_csv(f) for f in pbp_files]
            self.pbp = pd.concat(dfs, ignore_index=True)
            print(f"  Loaded {len(self.pbp)} plays from PBP data")

        # Schedules
        sched_file = RAW_DIR / "schedules.csv"
        if sched_file.exists():
            self.schedules = pd.read_csv(sched_file)
            print(f"  Loaded {len(self.schedules)} games from schedules")

        # Team EPA (processed)
        epa_file = PROCESSED_DIR / "team_epa_weekly.csv"
        if epa_file.exists():
            self.team_epa = pd.read_csv(epa_file)
            print(f"  Loaded team EPA data")

        # QB stats (processed)
        qb_file = PROCESSED_DIR / "qb_stats_weekly.csv"
        if qb_file.exists():
            self.qb_stats = pd.read_csv(qb_file)
            print(f"  Loaded QB stats")

        # Elo ratings (processed) - merge unit_elos with team_elo
        unit_elo_file = PROCESSED_DIR / "unit_elos.csv"
        team_elo_file = PROCESSED_DIR / "team_elo.csv"

        if unit_elo_file.exists() and team_elo_file.exists():
            unit_elos = pd.read_csv(unit_elo_file)
            team_elos = pd.read_csv(team_elo_file)
            # Merge to get team_elo column
            self.elos = pd.merge(unit_elos, team_elos[['team', 'team_elo']], on='team', how='left')
            print(f"  Loaded Elo ratings")
        elif unit_elo_file.exists():
            self.elos = pd.read_csv(unit_elo_file)
            # Use combined_elo as team_elo fallback
            if 'combined_elo' in self.elos.columns and 'team_elo' not in self.elos.columns:
                self.elos['team_elo'] = self.elos['combined_elo']
            print(f"  Loaded Elo ratings (unit only)")

        # Determine current week if not set
        if self.current_week is None and self.schedules is not None:
            current_games = self.schedules[
                (self.schedules['season'] == self.current_season) &
                (self.schedules['home_score'].isna())
            ]
            if len(current_games) > 0:
                self.current_week = current_games['week'].min()
            else:
                self.current_week = self.schedules[
                    self.schedules['season'] == self.current_season
                ]['week'].max()
            print(f"  Current week detected: {self.current_week}")

    def get_teams(self) -> List[str]:
        """Get list of all NFL teams"""
        if self.schedules is not None:
            teams = set(self.schedules['home_team'].unique())
            teams.update(self.schedules['away_team'].unique())
            return sorted(list(teams))
        return []

    def calculate_rolling_stats(self, df: pd.DataFrame, team_col: str,
                                 stat_cols: List[str], n_games: int = 5) -> pd.DataFrame:
        """
        Calculate rolling averages for stats.
        Weighting: 60% recent (5 games) + 40% season
        """
        result = []

        for team in df[team_col].unique():
            team_data = df[df[team_col] == team].sort_values(['season', 'week'])

            # Get recent N games
            recent = team_data.tail(n_games)

            # Get season data
            current_season = team_data[team_data['season'] == self.current_season]

            for col in stat_cols:
                if col in team_data.columns:
                    recent_avg = recent[col].mean() if len(recent) > 0 else 0
                    season_avg = current_season[col].mean() if len(current_season) > 0 else recent_avg

                    # Weighted average: 60% recent + 40% season
                    weighted = 0.6 * recent_avg + 0.4 * season_avg

                    result.append({
                        'team': team,
                        f'{col}_recent': recent_avg,
                        f'{col}_season': season_avg,
                        f'{col}_weighted': weighted
                    })

        return pd.DataFrame(result)

    def calculate_qb_quality(self) -> pd.DataFrame:
        """
        Calculate QB Quality factor (30% weight)
        - EPA/dropback
        - CPOE (Completion % Over Expected)
        - Pressure-to-sack rate
        - Playoff wins (historical bonus)
        """
        print("\nCalculating QB Quality factors...")

        if self.qb_stats is None:
            print("  Warning: QB stats not available, using offense Elo as proxy")
            # Use offense Elo as a proxy for QB quality
            qb_proxy = []
            for team in self.get_teams():
                off_elo = 1500
                if self.elos is not None:
                    team_row = self.elos[self.elos['team'] == team]
                    if len(team_row) > 0 and 'offense_elo' in team_row.columns:
                        off_elo = team_row.iloc[0]['offense_elo']

                # Normalize offense Elo to 0-100 scale (typically 1350-1650)
                qb_quality = (off_elo - 1350) / 300 * 100
                qb_quality = np.clip(qb_quality, 0, 100)

                qb_proxy.append({
                    'team': team,
                    'primary_qb': 'N/A',
                    'qb_quality_score': qb_quality
                })
            return pd.DataFrame(qb_proxy)

        qb_factors = []

        for team in self.get_teams():
            team_qbs = self.qb_stats[self.qb_stats['team'] == team]

            if len(team_qbs) == 0:
                continue

            # Get primary QB (most dropbacks)
            primary_qb = team_qbs.groupby('qb_name')['dropbacks'].sum().idxmax()
            qb_data = team_qbs[team_qbs['qb_name'] == primary_qb]

            # Recent 5 games
            recent = qb_data.tail(5)

            # Calculate metrics
            epa_dropback = recent['epa_dropback'].mean() if len(recent) > 0 else 0
            cpoe = recent['cpoe'].mean() if 'cpoe' in recent.columns and len(recent) > 0 else 0

            # Pressure-to-sack rate (inverse is better)
            if 'sacks' in recent.columns and 'dropbacks' in recent.columns:
                sack_rate = recent['sacks'].sum() / max(recent['dropbacks'].sum(), 1)
            else:
                sack_rate = 0.05  # Default

            # Normalize to 0-100 scale
            # EPA/dropback typically ranges from -0.3 to +0.3
            epa_score = (epa_dropback + 0.3) / 0.6 * 100
            epa_score = np.clip(epa_score, 0, 100)

            # CPOE typically ranges from -5 to +5
            cpoe_score = (cpoe + 5) / 10 * 100
            cpoe_score = np.clip(cpoe_score, 0, 100)

            # Sack rate (lower is better) - typically 3-10%
            sack_score = (0.10 - sack_rate) / 0.07 * 100
            sack_score = np.clip(sack_score, 0, 100)

            # Composite QB Quality score
            qb_quality = (epa_score * 0.50) + (cpoe_score * 0.30) + (sack_score * 0.20)

            qb_factors.append({
                'team': team,
                'primary_qb': primary_qb,
                'epa_dropback': epa_dropback,
                'cpoe': cpoe,
                'sack_rate': sack_rate,
                'epa_score': epa_score,
                'cpoe_score': cpoe_score,
                'sack_score': sack_score,
                'qb_quality_score': qb_quality
            })

        return pd.DataFrame(qb_factors)

    def calculate_team_efficiency(self) -> pd.DataFrame:
        """
        Calculate Team Efficiency factor (25% weight)
        - Team Elo
        - Offensive EPA/play
        - Defensive EPA/play
        """
        print("\nCalculating Team Efficiency factors...")

        efficiency_factors = []

        for team in self.get_teams():
            # Get Elo ratings
            team_elo = 1500  # Default
            off_elo = 1500
            def_elo = 1500

            if self.elos is not None:
                team_row = self.elos[self.elos['team'] == team]
                if len(team_row) > 0:
                    team_elo = team_row.iloc[0]['team_elo']
                    off_elo = team_row.iloc[0]['offense_elo']
                    def_elo = team_row.iloc[0]['defense_elo']

            # Get EPA stats
            off_epa = 0
            def_epa = 0

            if self.team_epa is not None:
                team_epa = self.team_epa[
                    (self.team_epa['team'] == team) &
                    (self.team_epa['season'] == self.current_season)
                ]
                if len(team_epa) > 0:
                    recent = team_epa.tail(5)
                    off_epa = recent['off_epa_play'].mean() if 'off_epa_play' in recent.columns else 0
                    def_epa = recent['def_epa_play'].mean() if 'def_epa_play' in recent.columns else 0

            # Normalize Elo (typically 1300-1700)
            elo_score = (team_elo - 1300) / 400 * 100
            elo_score = np.clip(elo_score, 0, 100)

            # Normalize EPA (typically -0.2 to +0.2)
            off_epa_score = (off_epa + 0.2) / 0.4 * 100
            off_epa_score = np.clip(off_epa_score, 0, 100)

            # For defense, negative EPA is good
            def_epa_score = (0.2 - def_epa) / 0.4 * 100
            def_epa_score = np.clip(def_epa_score, 0, 100)

            # Composite efficiency score
            efficiency_score = (elo_score * 0.40) + (off_epa_score * 0.30) + (def_epa_score * 0.30)

            efficiency_factors.append({
                'team': team,
                'team_elo': team_elo,
                'offense_elo': off_elo,
                'defense_elo': def_elo,
                'off_epa_play': off_epa,
                'def_epa_play': def_epa,
                'elo_score': elo_score,
                'off_epa_score': off_epa_score,
                'def_epa_score': def_epa_score,
                'efficiency_score': efficiency_score
            })

        return pd.DataFrame(efficiency_factors)

    def calculate_line_play(self) -> pd.DataFrame:
        """
        Calculate Line Play factor (20% weight)
        - OL pass block win rate (estimated from sack/pressure data)
        - Pressure rate allowed
        - OL-WR gap (how much OL affects passing game)
        """
        print("\nCalculating Line Play factors...")

        line_factors = []

        if self.pbp is None:
            print("  Warning: PBP data not available")
            return pd.DataFrame()

        for team in self.get_teams():
            # Get team's offensive plays
            team_plays = self.pbp[
                (self.pbp['posteam'] == team) &
                (self.pbp['play_type'] == 'pass') &
                (self.pbp['season'] == self.current_season)
            ]

            if len(team_plays) == 0:
                continue

            # Calculate sack rate (proxy for OL performance)
            sacks = team_plays['sack'].sum() if 'sack' in team_plays.columns else 0
            dropbacks = len(team_plays)
            sack_rate = sacks / max(dropbacks, 1)

            # Calculate pressure proxy (sacks + incompletions on quick throws)
            pressure_proxy = sack_rate  # Simplified

            # OL pass block score (lower sack rate = better)
            # Typical sack rate: 4-10%
            ol_score = (0.10 - sack_rate) / 0.06 * 100
            ol_score = np.clip(ol_score, 0, 100)

            # Pressure score
            pressure_score = (0.10 - pressure_proxy) / 0.06 * 100
            pressure_score = np.clip(pressure_score, 0, 100)

            # Run blocking (yards before contact proxy)
            run_plays = self.pbp[
                (self.pbp['posteam'] == team) &
                (self.pbp['play_type'] == 'run') &
                (self.pbp['season'] == self.current_season)
            ]
            ypc = run_plays['yards_gained'].mean() if len(run_plays) > 0 else 4.0

            # Run blocking score (typical YPC: 3.5-5.0)
            run_score = (ypc - 3.5) / 1.5 * 100
            run_score = np.clip(run_score, 0, 100)

            # Composite line play score
            line_score = (ol_score * 0.40) + (pressure_score * 0.30) + (run_score * 0.30)

            line_factors.append({
                'team': team,
                'sack_rate': sack_rate,
                'pressure_rate': pressure_proxy,
                'yards_per_carry': ypc,
                'ol_pass_score': ol_score,
                'pressure_score': pressure_score,
                'run_block_score': run_score,
                'line_play_score': line_score
            })

        return pd.DataFrame(line_factors)

    def calculate_situational(self) -> pd.DataFrame:
        """
        Calculate Situational factor (15% weight)
        - 3rd down success rate
        - Red zone TD rate
        - 4th quarter performance
        """
        print("\nCalculating Situational factors...")

        situational_factors = []

        if self.pbp is None:
            print("  Warning: PBP data not available")
            return pd.DataFrame()

        for team in self.get_teams():
            team_plays = self.pbp[
                (self.pbp['posteam'] == team) &
                (self.pbp['season'] == self.current_season)
            ]

            if len(team_plays) == 0:
                continue

            # 3rd down success rate
            third_downs = team_plays[team_plays['down'] == 3]
            third_down_success = third_downs['success'].mean() if len(third_downs) > 0 else 0.35

            # Red zone (inside 20)
            red_zone = team_plays[
                (team_plays['yardline_100'].notna()) &
                (team_plays['yardline_100'] <= 20)
            ]
            rz_tds = red_zone['touchdown'].sum() if len(red_zone) > 0 and 'touchdown' in red_zone.columns else 0
            rz_attempts = len(red_zone[red_zone['play_type'].isin(['pass', 'run'])]) if len(red_zone) > 0 else 1
            rz_td_rate = rz_tds / max(rz_attempts, 1)

            # 4th quarter EPA
            q4_plays = team_plays[team_plays['qtr'] == 4]
            q4_epa = q4_plays['epa'].mean() if len(q4_plays) > 0 else 0

            # Normalize scores (typical ranges)
            # 3rd down: 30-50%
            third_score = (third_down_success - 0.30) / 0.20 * 100
            third_score = np.clip(third_score, 0, 100)

            # RZ TD rate: 40-70%
            rz_score = (rz_td_rate - 0.40) / 0.30 * 100
            rz_score = np.clip(rz_score, 0, 100)

            # Q4 EPA: -0.1 to +0.1
            q4_score = (q4_epa + 0.1) / 0.2 * 100
            q4_score = np.clip(q4_score, 0, 100)

            # Composite situational score
            situational_score = (third_score * 0.35) + (rz_score * 0.35) + (q4_score * 0.30)

            situational_factors.append({
                'team': team,
                'third_down_success': third_down_success,
                'red_zone_td_rate': rz_td_rate,
                'q4_epa': q4_epa,
                'third_score': third_score,
                'rz_score': rz_score,
                'q4_score': q4_score,
                'situational_score': situational_score
            })

        return pd.DataFrame(situational_factors)

    def calculate_luck_regression(self) -> pd.DataFrame:
        """
        Calculate Luck Regression factor (10% weight)
        - Interceptable passes not intercepted
        - INT luck factor
        - Fumble recovery luck
        - Close game record vs expected
        """
        print("\nCalculating Luck Regression factors...")

        luck_factors = []

        if self.pbp is None:
            print("  Warning: PBP data not available")
            return pd.DataFrame()

        for team in self.get_teams():
            # Offensive plays
            off_plays = self.pbp[
                (self.pbp['posteam'] == team) &
                (self.pbp['season'] == self.current_season) &
                (self.pbp['play_type'] == 'pass')
            ]

            # Defensive plays against
            def_plays = self.pbp[
                (self.pbp['defteam'] == team) &
                (self.pbp['season'] == self.current_season) &
                (self.pbp['play_type'] == 'pass')
            ]

            # INT rate on offense (lower = better for team)
            off_ints = off_plays['interception'].sum() if len(off_plays) > 0 else 0
            off_passes = len(off_plays)
            off_int_rate = off_ints / max(off_passes, 1)

            # INT rate on defense (higher = better for team)
            def_ints = def_plays['interception'].sum() if len(def_plays) > 0 else 0
            def_passes = len(def_plays)
            def_int_rate = def_ints / max(def_passes, 1)

            # Expected INT rates (league average is ~2.5%)
            expected_int_rate = 0.025

            # Luck factor: actual vs expected
            # Positive = lucky (fewer INTs thrown than expected, more INTs caught than expected)
            off_luck = expected_int_rate - off_int_rate  # Positive = lucky
            def_luck = def_int_rate - expected_int_rate  # Positive = lucky

            # Fumble luck (simplified)
            fumbles = self.pbp[
                (self.pbp['posteam'] == team) &
                (self.pbp['season'] == self.current_season) &
                (self.pbp['fumble_lost'] == 1)
            ] if 'fumble_lost' in self.pbp.columns else pd.DataFrame()

            fumble_rate = len(fumbles) / max(off_passes, 1)
            expected_fumble_rate = 0.01  # ~1% of plays
            fumble_luck = expected_fumble_rate - fumble_rate

            # Total luck factor (will regress toward 0)
            total_luck = off_luck + def_luck + fumble_luck

            # Regression adjustment: teams with positive luck should regress down
            # Teams with negative luck should regress up
            # Score: how much upside from luck regression
            # Negative luck = positive regression potential = higher score
            regression_score = 50 - (total_luck * 1000)  # Scale for readability
            regression_score = np.clip(regression_score, 0, 100)

            luck_factors.append({
                'team': team,
                'off_int_rate': off_int_rate,
                'def_int_rate': def_int_rate,
                'off_luck': off_luck,
                'def_luck': def_luck,
                'fumble_luck': fumble_luck,
                'total_luck': total_luck,
                'luck_regression_score': regression_score
            })

        return pd.DataFrame(luck_factors)

    def calculate_all_factors(self) -> pd.DataFrame:
        """Calculate all factors and combine into final factor scores"""
        print("=" * 60)
        print("FACTOR CALCULATION")
        print("=" * 60)

        self.load_data()

        # Calculate each factor
        qb_quality = self.calculate_qb_quality()
        efficiency = self.calculate_team_efficiency()
        line_play = self.calculate_line_play()
        situational = self.calculate_situational()
        luck = self.calculate_luck_regression()

        # Merge all factors
        all_factors = efficiency[['team', 'efficiency_score']].copy()

        if len(qb_quality) > 0:
            all_factors = all_factors.merge(
                qb_quality[['team', 'qb_quality_score', 'primary_qb']],
                on='team', how='left'
            )

        if len(line_play) > 0:
            all_factors = all_factors.merge(
                line_play[['team', 'line_play_score']],
                on='team', how='left'
            )

        if len(situational) > 0:
            all_factors = all_factors.merge(
                situational[['team', 'situational_score']],
                on='team', how='left'
            )

        if len(luck) > 0:
            all_factors = all_factors.merge(
                luck[['team', 'luck_regression_score']],
                on='team', how='left'
            )

        # Fill NaN with 50 (average)
        score_cols = ['qb_quality_score', 'efficiency_score', 'line_play_score',
                      'situational_score', 'luck_regression_score']
        for col in score_cols:
            if col in all_factors.columns:
                all_factors[col] = all_factors[col].fillna(50)

        # Save individual factor dataframes
        if len(qb_quality) > 0:
            qb_quality.to_csv(PROCESSED_DIR / "factor_qb_quality.csv", index=False)
        if len(efficiency) > 0:
            efficiency.to_csv(PROCESSED_DIR / "factor_efficiency.csv", index=False)
        if len(line_play) > 0:
            line_play.to_csv(PROCESSED_DIR / "factor_line_play.csv", index=False)
        if len(situational) > 0:
            situational.to_csv(PROCESSED_DIR / "factor_situational.csv", index=False)
        if len(luck) > 0:
            luck.to_csv(PROCESSED_DIR / "factor_luck.csv", index=False)

        # Save combined factors
        all_factors.to_csv(PROCESSED_DIR / "team_factors.csv", index=False)
        print(f"\nSaved team_factors.csv with {len(all_factors)} teams")

        # Print summary
        print("\n" + "=" * 60)
        print("FACTOR SUMMARY (Top 10 by Efficiency)")
        print("=" * 60)
        print(all_factors.sort_values('efficiency_score', ascending=False).head(10).to_string(index=False))

        return all_factors


if __name__ == "__main__":
    calculator = FactorCalculator(current_season=2024)
    factors = calculator.calculate_all_factors()
