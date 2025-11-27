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
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


class CurrentRosterFetcher:
    """Fetches current NFL roster/starter information from ESPN API"""

    ESPN_TEAM_IDS = {
        'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3,
        'CIN': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
        'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12, 'LV': 13, 'LAC': 24,
        'LAR': 14, 'MIA': 15, 'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
        'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SF': 25, 'SEA': 26, 'TB': 27,
        'TEN': 10, 'WAS': 28
    }

    # Position groups we care about
    OFFENSE_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'OT', 'OG', 'C', 'T', 'G']
    DEFENSE_POSITIONS = ['DE', 'DT', 'LB', 'CB', 'S', 'NT', 'OLB', 'ILB', 'MLB', 'FS', 'SS']
    KEY_POSITIONS = ['QB', 'WR', 'RB', 'TE', 'DE', 'CB', 'LB', 'S']

    def __init__(self):
        self._cache = {}  # team -> {position: [players]}
        self._roster_cache = {}  # team -> full roster data

    def get_team_roster(self, team_abbrev: str) -> Dict:
        """Get full roster for a team, organized by position"""
        if team_abbrev in self._roster_cache:
            return self._roster_cache[team_abbrev]

        team_id = self.ESPN_TEAM_IDS.get(team_abbrev)
        if not team_id:
            return {}

        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            roster = {}
            for group in data.get('athletes', []):
                for player in group.get('items', []):
                    pos = player.get('position', {}).get('abbreviation', '')
                    name = player.get('displayName', '')
                    if pos and name:
                        if pos not in roster:
                            roster[pos] = []
                        roster[pos].append({
                            'name': name,
                            'id': player.get('id'),
                            'jersey': player.get('jersey'),
                            'status': player.get('status', {}).get('type', 'active')
                        })

            self._roster_cache[team_abbrev] = roster
            return roster

        except Exception as e:
            print(f"  Warning: Could not fetch roster for {team_abbrev}: {e}")
            return {}

    def get_current_qb(self, team_abbrev: str) -> Optional[str]:
        """Get the current starting QB for a team"""
        roster = self.get_team_roster(team_abbrev)
        qbs = roster.get('QB', [])
        # Return first QB (typically the starter)
        return qbs[0]['name'] if qbs else None

    def get_position_starters(self, team_abbrev: str, position: str, count: int = 1) -> List[str]:
        """Get current starters at a position (e.g., top 2 WRs)"""
        roster = self.get_team_roster(team_abbrev)
        players = roster.get(position, [])
        return [p['name'] for p in players[:count]]

    def get_all_current_qbs(self) -> Dict[str, str]:
        """Get current QBs for all 32 teams"""
        print("  Fetching current starting QBs from ESPN roster...")
        qbs = {}
        for team in self.ESPN_TEAM_IDS.keys():
            qb = self.get_current_qb(team)
            if qb:
                qbs[team] = qb
        print(f"  Found current QBs for {len(qbs)} teams")
        return qbs

    def get_all_starters(self) -> Dict[str, Dict[str, List[str]]]:
        """Get key starters for all 32 teams"""
        print("  Fetching current rosters from ESPN...")
        all_starters = {}

        for team in self.ESPN_TEAM_IDS.keys():
            roster = self.get_team_roster(team)
            starters = {
                'QB': self.get_position_starters(team, 'QB', 1),
                'RB': self.get_position_starters(team, 'RB', 2),
                'WR': self.get_position_starters(team, 'WR', 3),
                'TE': self.get_position_starters(team, 'TE', 1),
                'OL': (self.get_position_starters(team, 'OT', 2) +
                       self.get_position_starters(team, 'OG', 2) +
                       self.get_position_starters(team, 'C', 1) +
                       self.get_position_starters(team, 'T', 2) +
                       self.get_position_starters(team, 'G', 2)),
                'DE': self.get_position_starters(team, 'DE', 2),
                'DT': self.get_position_starters(team, 'DT', 2),
                'LB': (self.get_position_starters(team, 'LB', 3) +
                       self.get_position_starters(team, 'OLB', 2) +
                       self.get_position_starters(team, 'ILB', 2) +
                       self.get_position_starters(team, 'MLB', 1)),
                'CB': self.get_position_starters(team, 'CB', 2),
                'S': (self.get_position_starters(team, 'S', 2) +
                      self.get_position_starters(team, 'FS', 1) +
                      self.get_position_starters(team, 'SS', 1)),
            }
            all_starters[team] = starters

        teams_with_data = sum(1 for t in all_starters if any(all_starters[t].values()))
        print(f"  Found roster data for {teams_with_data} teams")
        return all_starters

    def get_position_count(self, team_abbrev: str, position: str) -> int:
        """Get count of players at a position (useful for depth assessment)"""
        roster = self.get_team_roster(team_abbrev)
        return len(roster.get(position, []))


class FactorCalculator:
    """Calculate all factors for NFL teams"""

    def __init__(self, current_season: int = 2024, current_week: int = None):
        self.current_season = current_season
        self.current_week = current_week
        self.pbp = None
        self.schedules = None
        self.team_epa = None
        self.qb_stats = None
        self.weekly_stats = None  # Player-level weekly stats (WR, RB, TE, QB)
        self.elos = None
        self.roster_fetcher = CurrentRosterFetcher()
        self.current_qbs = None  # Will be populated with current starting QBs
        self.current_starters = None  # All key starters by position

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

        # Weekly player stats (WR, RB, TE, QB)
        weekly_file = RAW_DIR / "weekly_stats.csv"
        if weekly_file.exists():
            self.weekly_stats = pd.read_csv(weekly_file)
            print(f"  Loaded weekly player stats ({len(self.weekly_stats)} rows)")

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

    def _match_player_name(self, full_name: str) -> str:
        """Convert full name to first initial + last name pattern for matching.
        E.g., 'Tyreek Hill' -> 't.hill'
        """
        parts = full_name.split()
        if len(parts) >= 2:
            first_initial = parts[0][0].lower()
            last_name = parts[-1].lower()
            return f"{first_initial}.{last_name}"
        return full_name.lower()

    def _get_player_stats(self, player_name: str, position: str) -> Optional[pd.DataFrame]:
        """Look up player stats from weekly_stats by name and position.
        Searches across ALL teams since player may have been traded.
        Returns most recent 5 games of stats.
        """
        if self.weekly_stats is None:
            return None

        pattern = self._match_player_name(player_name)

        # Match by player_name column (which uses short format like "T.Hill")
        position_stats = self.weekly_stats[self.weekly_stats['position'] == position]
        player_data = position_stats[position_stats['player_name'].str.lower() == pattern]

        if len(player_data) == 0:
            # Try matching against display name
            player_data = position_stats[
                position_stats['player_display_name'].str.lower() == player_name.lower()
            ]

        if len(player_data) == 0:
            return None

        # Return most recent 5 games
        return player_data.sort_values(['season', 'week'], ascending=False).head(5)

    def calculate_receiver_quality(self, team: str) -> Dict[str, Any]:
        """Calculate receiving corps quality based on current WR/TE starters.
        Returns dict with receiver metrics.
        """
        if self.weekly_stats is None:
            return {'wr_epa': 0, 'wr_target_share': 0, 'wr_score': 50}

        # Get current WR and TE starters from roster
        if self.current_starters is None:
            self.current_starters = self.roster_fetcher.get_all_starters()

        team_starters = self.current_starters.get(team, {})
        wr_starters = team_starters.get('WR', [])
        te_starters = team_starters.get('TE', [])

        all_receivers = wr_starters[:3] + te_starters[:1]  # Top 3 WRs + TE1

        total_epa = 0
        total_target_share = 0
        players_found = 0
        receiver_names = []

        for receiver in all_receivers:
            # Try WR first, then TE
            stats = self._get_player_stats(receiver, 'WR')
            if stats is None:
                stats = self._get_player_stats(receiver, 'TE')

            if stats is not None and len(stats) > 0:
                epa = stats['receiving_epa'].mean() if 'receiving_epa' in stats.columns else 0
                target_share = stats['target_share'].mean() if 'target_share' in stats.columns else 0

                # Handle NaN values
                epa = 0 if pd.isna(epa) else epa
                target_share = 0 if pd.isna(target_share) else target_share

                total_epa += epa
                total_target_share += target_share
                players_found += 1
                receiver_names.append(receiver)

        if players_found == 0:
            return {'wr_epa': 0, 'wr_target_share': 0, 'wr_score': 50, 'receivers': []}

        avg_epa = total_epa / players_found
        avg_target_share = total_target_share / players_found

        # Normalize to 0-100 score
        # Receiving EPA typically ranges from -0.5 to +0.5
        epa_score = (avg_epa + 0.5) / 1.0 * 100
        epa_score = np.clip(epa_score, 0, 100)

        # Target share typically 0.05 to 0.25
        target_score = (avg_target_share - 0.05) / 0.20 * 100
        target_score = np.clip(target_score, 0, 100)

        # Composite score
        wr_score = (epa_score * 0.6) + (target_score * 0.4)

        return {
            'wr_epa': avg_epa,
            'wr_target_share': avg_target_share,
            'wr_score': wr_score,
            'receivers': receiver_names
        }

    def calculate_rusher_quality(self, team: str) -> Dict[str, Any]:
        """Calculate rushing corps quality based on current RB starters.
        Returns dict with rushing metrics.
        """
        if self.weekly_stats is None:
            return {'rb_epa': 0, 'rb_ypc': 0, 'rb_score': 50}

        # Get current RB starters from roster
        if self.current_starters is None:
            self.current_starters = self.roster_fetcher.get_all_starters()

        team_starters = self.current_starters.get(team, {})
        rb_starters = team_starters.get('RB', [])[:2]  # Top 2 RBs

        total_epa = 0
        total_ypc = 0
        players_found = 0
        rusher_names = []

        for rusher in rb_starters:
            stats = self._get_player_stats(rusher, 'RB')

            if stats is not None and len(stats) > 0:
                epa = stats['rushing_epa'].mean() if 'rushing_epa' in stats.columns else 0
                carries = stats['carries'].sum() if 'carries' in stats.columns else 0
                yards = stats['rushing_yards'].sum() if 'rushing_yards' in stats.columns else 0

                # Handle NaN values
                epa = 0 if pd.isna(epa) else epa
                ypc = yards / max(carries, 1) if carries > 0 else 4.0

                total_epa += epa
                total_ypc += ypc
                players_found += 1
                rusher_names.append(rusher)

        if players_found == 0:
            return {'rb_epa': 0, 'rb_ypc': 4.0, 'rb_score': 50, 'rushers': []}

        avg_epa = total_epa / players_found
        avg_ypc = total_ypc / players_found

        # Normalize to 0-100 score
        # Rushing EPA typically ranges from -0.3 to +0.2
        epa_score = (avg_epa + 0.3) / 0.5 * 100
        epa_score = np.clip(epa_score, 0, 100)

        # YPC typically 3.5 to 5.5
        ypc_score = (avg_ypc - 3.5) / 2.0 * 100
        ypc_score = np.clip(ypc_score, 0, 100)

        # Composite score
        rb_score = (epa_score * 0.5) + (ypc_score * 0.5)

        return {
            'rb_epa': avg_epa,
            'rb_ypc': avg_ypc,
            'rb_score': rb_score,
            'rushers': rusher_names
        }

    def calculate_qb_quality(self) -> pd.DataFrame:
        """
        Calculate QB Quality factor (30% weight)
        - EPA/dropback
        - CPOE (Completion % Over Expected)
        - Pressure-to-sack rate
        - Playoff wins (historical bonus)

        Uses CURRENT starting QB from ESPN roster API, not historical leader.
        """
        print("\nCalculating QB Quality factors...")

        # Fetch current starting QBs from ESPN roster
        if self.current_qbs is None:
            self.current_qbs = self.roster_fetcher.get_all_current_qbs()

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

                # Get current QB name from roster if available
                current_qb = self.current_qbs.get(team, 'N/A')

                qb_proxy.append({
                    'team': team,
                    'primary_qb': current_qb,
                    'qb_quality_score': qb_quality
                })
            return pd.DataFrame(qb_proxy)

        qb_factors = []

        for team in self.get_teams():
            team_qbs = self.qb_stats[self.qb_stats['team'] == team]

            if len(team_qbs) == 0:
                continue

            # First check for current starting QB from roster API
            current_qb = self.current_qbs.get(team)
            qb_data = None

            if current_qb:
                # Try to find current starter in QB stats
                # Match by first initial + last name (e.g., "Jacoby Brissett" -> "J.Brissett")
                parts = current_qb.split()
                if len(parts) >= 2:
                    first_initial = parts[0][0].upper()
                    last_name = parts[-1]
                    # Build pattern like "J.Brissett" or just last name as fallback
                    exact_pattern = f"{first_initial}.{last_name}".lower()
                else:
                    exact_pattern = current_qb.lower()
                    last_name = current_qb

                # First check team-specific stats with exact match (first initial + last name)
                matching_qbs = team_qbs[team_qbs['qb_name'].str.lower() == exact_pattern]

                if len(matching_qbs) > 0:
                    primary_qb = matching_qbs.groupby('qb_name')['dropbacks'].sum().idxmax()
                    qb_data = matching_qbs[matching_qbs['qb_name'] == primary_qb]
                    print(f"  {team}: Using current starter {current_qb} -> {primary_qb} (team stats)")
                else:
                    # Search in ALL teams with exact match - QB may have played elsewhere
                    all_matching = self.qb_stats[self.qb_stats['qb_name'].str.lower() == exact_pattern]

                    if len(all_matching) > 0:
                        primary_qb = all_matching.groupby('qb_name')['dropbacks'].sum().idxmax()
                        qb_data = all_matching[all_matching['qb_name'] == primary_qb]
                        prev_teams = all_matching['team'].unique()
                        print(f"  {team}: Using current starter {current_qb} -> {primary_qb} (from {', '.join(prev_teams)})")
                    else:
                        # Current starter has no NFL stats (likely a rookie)
                        # Use the current starter's name and assign a baseline rookie score
                        parts = current_qb.split()
                        if len(parts) >= 2:
                            primary_qb = f"{parts[0][0]}.{parts[-1]}"
                        else:
                            primary_qb = current_qb
                        qb_data = None  # Mark as no stats - will use baseline
                        print(f"  {team}: Current starter {current_qb} -> {primary_qb} (ROOKIE - no NFL stats)")
            else:
                # Fallback to historical leader if roster API failed
                primary_qb = team_qbs.groupby('qb_name')['dropbacks'].sum().idxmax()
                qb_data = team_qbs[team_qbs['qb_name'] == primary_qb]

            # Handle rookie QBs with no NFL stats
            if qb_data is None or len(qb_data) == 0:
                # Rookie/unknown QB - assign baseline score (below average)
                qb_factors.append({
                    'team': team,
                    'primary_qb': primary_qb,
                    'epa_dropback': 0,
                    'cpoe': 0,
                    'sack_rate': 0.06,  # Average sack rate
                    'epa_score': 30,    # Below average
                    'cpoe_score': 30,   # Below average
                    'sack_score': 50,   # Average
                    'qb_quality_score': 35  # Below average (unknown)
                })
                continue

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

    def calculate_skill_positions(self) -> pd.DataFrame:
        """
        Calculate Skill Position quality using CURRENT starters from ESPN roster.
        - WR corps quality (based on current WRs)
        - RB corps quality (based on current RBs)

        This adjusts the offensive potential based on who is actually playing.
        """
        print("\nCalculating Skill Position factors (using current starters)...")

        # Fetch all starters if not already done
        if self.current_starters is None:
            self.current_starters = self.roster_fetcher.get_all_starters()

        skill_factors = []

        for team in self.get_teams():
            # Get receiver quality
            wr_data = self.calculate_receiver_quality(team)
            rb_data = self.calculate_rusher_quality(team)

            receivers = wr_data.get('receivers', [])
            rushers = rb_data.get('rushers', [])

            if receivers or rushers:
                print(f"  {team}: WRs={receivers[:2]}, RBs={rushers[:1]}")

            # Combine into skill position score
            wr_score = wr_data.get('wr_score', 50)
            rb_score = rb_data.get('rb_score', 50)

            # Weight WRs slightly more in modern NFL
            skill_score = (wr_score * 0.6) + (rb_score * 0.4)

            skill_factors.append({
                'team': team,
                'wr_epa': wr_data.get('wr_epa', 0),
                'wr_target_share': wr_data.get('wr_target_share', 0),
                'wr_score': wr_score,
                'rb_epa': rb_data.get('rb_epa', 0),
                'rb_ypc': rb_data.get('rb_ypc', 4.0),
                'rb_score': rb_score,
                'skill_position_score': skill_score,
                'current_wrs': ', '.join(receivers[:3]) if receivers else 'N/A',
                'current_rbs': ', '.join(rushers[:2]) if rushers else 'N/A'
            })

        return pd.DataFrame(skill_factors)

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
        skill_positions = self.calculate_skill_positions()  # NEW: WR/RB based on current starters
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

        if len(skill_positions) > 0:
            all_factors = all_factors.merge(
                skill_positions[['team', 'skill_position_score', 'wr_score', 'rb_score',
                                'current_wrs', 'current_rbs']],
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
                      'skill_position_score', 'wr_score', 'rb_score',
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
        if len(skill_positions) > 0:
            skill_positions.to_csv(PROCESSED_DIR / "factor_skill_positions.csv", index=False)
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
