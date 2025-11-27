"""
NFL Starter Detection & Injury Integration Module
==================================================
Ensures model uses stats from CURRENT healthy starters, not historical averages.

Data Sources:
- ESPN Depth Chart API: Declared starters
- ESPN Injuries API: Injury status (OUT, DOUBTFUL, QUESTIONABLE)
- nfl_data_py: Play-by-play verification of actual starters

Usage:
    detector = StarterDetector()
    starters = detector.get_all_starters()
    # Returns dict with current QB for each team + confidence level
"""

import requests
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import time


OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class StarterDetector:
    """Detects current NFL starters accounting for injuries and depth chart changes."""

    # ESPN Team ID mapping
    ESPN_TEAM_IDS = {
        'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3,
        'CIN': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
        'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12, 'LV': 13, 'LAC': 24,
        'LAR': 14, 'MIA': 15, 'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
        'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SF': 25, 'SEA': 26, 'TB': 27,
        'TEN': 10, 'WAS': 28
    }

    # Reverse mapping
    ESPN_ID_TO_TEAM = {v: k for k, v in ESPN_TEAM_IDS.items()}

    def __init__(self, cache_duration_hours: int = 4):
        """
        Initialize detector.

        Args:
            cache_duration_hours: How long to cache API responses
        """
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._depth_chart_cache = {}
        self._injury_cache = {}
        self._cache_timestamps = {}

    def get_all_starters(self, position: str = 'QB') -> Dict[str, dict]:
        """
        Get current starters for all 32 teams.

        Args:
            position: Position to check (default 'QB')

        Returns:
            Dict mapping team abbrev to starter info:
            {
                'ARI': {
                    'starter': 'Kyler Murray',
                    'backup': 'Clayton Tune',
                    'status': 'HEALTHY',  # or OUT, DOUBTFUL, QUESTIONABLE
                    'actual_starter': 'Kyler Murray',  # from PBP verification
                    'confidence': 'high',  # high, medium, low
                    'notes': [],
                    'games_started': 10,
                    'last_updated': '2025-11-26T15:30:00'
                },
                ...
            }
        """
        starters = {}

        for team_abbrev in self.ESPN_TEAM_IDS.keys():
            try:
                starters[team_abbrev] = self.get_team_starter(team_abbrev, position)
            except Exception as e:
                starters[team_abbrev] = {
                    'starter': 'UNKNOWN',
                    'backup': 'UNKNOWN',
                    'status': 'ERROR',
                    'confidence': 'none',
                    'notes': [f'Error fetching data: {str(e)}'],
                    'last_updated': datetime.now().isoformat()
                }

        return starters

    def get_team_starter(self, team_abbrev: str, position: str = 'QB') -> dict:
        """
        Get current starter for a specific team and position.

        Args:
            team_abbrev: Team abbreviation (e.g., 'ARI', 'KC')
            position: Position to check

        Returns:
            Starter info dict
        """
        team_id = self.ESPN_TEAM_IDS.get(team_abbrev)
        if not team_id:
            raise ValueError(f"Unknown team: {team_abbrev}")

        notes = []

        # Step 1: Get depth chart
        depth_chart = self._get_depth_chart(team_id)
        position_depth = depth_chart.get(position, [])

        if not position_depth:
            return {
                'starter': 'UNKNOWN',
                'backup': 'UNKNOWN',
                'status': 'NO_DATA',
                'confidence': 'none',
                'notes': [f'No {position} found in depth chart'],
                'last_updated': datetime.now().isoformat()
            }

        qb1_name = position_depth[0] if len(position_depth) > 0 else 'UNKNOWN'
        qb2_name = position_depth[1] if len(position_depth) > 1 else 'UNKNOWN'

        # Step 2: Check injuries
        injuries = self._get_injuries(team_id)
        qb1_status = injuries.get(qb1_name, 'HEALTHY')
        qb2_status = injuries.get(qb2_name, 'HEALTHY')

        # Step 3: Determine actual starter based on injury status
        if qb1_status == 'OUT':
            actual_starter = qb2_name
            confidence = 'high'
            notes.append(f'{qb1_name} is OUT - {qb2_name} starting')

            # Check if backup is also injured
            if qb2_status == 'OUT':
                qb3_name = position_depth[2] if len(position_depth) > 2 else 'UNKNOWN'
                actual_starter = qb3_name
                notes.append(f'{qb2_name} also OUT - {qb3_name} starting')
                confidence = 'medium'

        elif qb1_status == 'DOUBTFUL':
            actual_starter = qb1_name  # Listed but likely won't play
            confidence = 'low'
            notes.append(f'{qb1_name} is DOUBTFUL - may not play')

        elif qb1_status == 'QUESTIONABLE':
            actual_starter = qb1_name
            confidence = 'medium'
            notes.append(f'{qb1_name} is QUESTIONABLE')

        else:
            actual_starter = qb1_name
            confidence = 'high'

        # Step 4: Try to verify with PBP data (most recent game)
        pbp_starter = self._verify_with_pbp(team_abbrev)
        if pbp_starter and pbp_starter != actual_starter:
            notes.append(f'PBP shows {pbp_starter} started last game, not {actual_starter}')
            # Trust PBP over depth chart for recent history
            if confidence == 'high':
                confidence = 'medium'

        # Step 5: Get games started count
        games_started = self._get_games_started(team_abbrev, actual_starter)

        return {
            'starter': qb1_name,
            'backup': qb2_name,
            'status': qb1_status,
            'actual_starter': actual_starter,
            'confidence': confidence,
            'notes': notes,
            'games_started': games_started,
            'last_updated': datetime.now().isoformat()
        }

    def _get_depth_chart(self, team_id: int) -> Dict[str, List[str]]:
        """Fetch depth chart from ESPN API."""
        cache_key = f'depth_{team_id}'

        # Check cache
        if cache_key in self._depth_chart_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < self.cache_duration:
                return self._depth_chart_cache[cache_key]

        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/depthcharts"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching depth chart for team {team_id}: {e}")
            return {}

        # Parse depth chart
        depth_chart = {}

        try:
            items = data.get('items', [])
            for item in items:
                # Find offense positions
                positions = item.get('positions', {})
                for pos_data in positions.values():
                    pos_name = pos_data.get('position', {}).get('abbreviation', '')
                    athletes = pos_data.get('athletes', [])

                    player_names = []
                    for athlete in athletes:
                        name = athlete.get('athlete', {}).get('displayName', '')
                        if name:
                            player_names.append(name)

                    if pos_name and player_names:
                        depth_chart[pos_name] = player_names

        except Exception as e:
            print(f"Error parsing depth chart: {e}")

        # Cache result
        self._depth_chart_cache[cache_key] = depth_chart
        self._cache_timestamps[cache_key] = datetime.now()

        return depth_chart

    def _get_injuries(self, team_id: int) -> Dict[str, str]:
        """Fetch injury report from ESPN API."""
        cache_key = f'injuries_{team_id}'

        # Check cache
        if cache_key in self._injury_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < self.cache_duration:
                return self._injury_cache[cache_key]

        url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching injuries for team {team_id}: {e}")
            return {}

        injuries = {}

        try:
            items = data.get('items', [])
            for item in items:
                # Each item has a $ref to athlete details
                athlete_ref = item.get('athlete', {}).get('$ref', '')
                status = item.get('status', 'UNKNOWN')

                # Fetch athlete name if needed
                if athlete_ref:
                    try:
                        athlete_resp = requests.get(athlete_ref, timeout=5)
                        athlete_data = athlete_resp.json()
                        name = athlete_data.get('displayName', '')
                        if name:
                            # Normalize status
                            status_upper = status.upper()
                            if 'OUT' in status_upper:
                                injuries[name] = 'OUT'
                            elif 'DOUBT' in status_upper:
                                injuries[name] = 'DOUBTFUL'
                            elif 'QUESTION' in status_upper:
                                injuries[name] = 'QUESTIONABLE'
                            elif 'PROB' in status_upper:
                                injuries[name] = 'PROBABLE'
                            else:
                                injuries[name] = status_upper
                    except:
                        pass

        except Exception as e:
            print(f"Error parsing injuries: {e}")

        # Cache result
        self._injury_cache[cache_key] = injuries
        self._cache_timestamps[cache_key] = datetime.now()

        return injuries

    def _verify_with_pbp(self, team_abbrev: str, weeks_back: int = 1) -> Optional[str]:
        """
        Verify starter using play-by-play data.
        Returns the QB who had most pass attempts in most recent game.
        """
        try:
            # Get current week
            current_week = self._get_current_week()
            target_week = max(1, current_week - weeks_back)

            # Load PBP data
            pbp = nfl.import_pbp_data([2025])

            # Filter to team's most recent game
            team_plays = pbp[
                (pbp['posteam'] == team_abbrev) &
                (pbp['week'] == target_week) &
                (pbp['play_type'] == 'pass')
            ]

            if team_plays.empty:
                return None

            # Find QB with most attempts
            qb_attempts = team_plays.groupby('passer_player_name').size()
            if qb_attempts.empty:
                return None

            starter = qb_attempts.idxmax()
            return starter

        except Exception as e:
            print(f"Error verifying with PBP: {e}")
            return None

    def _get_games_started(self, team_abbrev: str, player_name: str) -> int:
        """Get number of games a player has started this season."""
        try:
            pbp = nfl.import_pbp_data([2025])

            # Count weeks where this player had majority of pass attempts
            team_plays = pbp[
                (pbp['posteam'] == team_abbrev) &
                (pbp['play_type'] == 'pass')
            ]

            games_started = 0
            for week in team_plays['week'].unique():
                week_plays = team_plays[team_plays['week'] == week]
                if week_plays.empty:
                    continue

                qb_attempts = week_plays.groupby('passer_player_name').size()
                if qb_attempts.empty:
                    continue

                week_starter = qb_attempts.idxmax()
                if week_starter == player_name:
                    games_started += 1

            return games_started

        except Exception as e:
            print(f"Error getting games started: {e}")
            return 0

    def _get_current_week(self) -> int:
        """Estimate current NFL week based on date."""
        # 2025 season starts Sept 4, 2025
        season_start = datetime(2025, 9, 4)
        today = datetime.now()

        if today < season_start:
            return 0

        days_since_start = (today - season_start).days
        current_week = (days_since_start // 7) + 1

        return min(current_week, 18)  # Cap at week 18


class QBStatsAdjuster:
    """Adjusts QB stats based on current starter information."""

    def __init__(self, starter_detector: StarterDetector):
        self.detector = starter_detector
        self._weekly_data = None
        self._seasonal_data = None

    def load_data(self):
        """Load weekly and seasonal QB data."""
        print("Loading QB data from nfl_data_py...")
        self._weekly_data = nfl.import_weekly_data([2024, 2025])
        self._seasonal_data = nfl.import_seasonal_data([2024, 2025])
        print(f"Loaded {len(self._weekly_data)} weekly records")

    def get_current_qb_stats(self, team_abbrev: str) -> dict:
        """
        Get stats for the CURRENT starting QB only.

        Returns:
            Dict with QB stats adjusted for current starter
        """
        if self._weekly_data is None:
            self.load_data()

        # Get current starter info
        starter_info = self.detector.get_team_starter(team_abbrev)
        current_qb = starter_info['actual_starter']
        confidence = starter_info['confidence']
        games_started = starter_info['games_started']

        # Filter weekly data to current QB
        qb_weekly = self._weekly_data[
            (self._weekly_data['player_display_name'] == current_qb) &
            (self._weekly_data['recent_team'] == team_abbrev) &
            (self._weekly_data['season'] == 2025)
        ]

        if qb_weekly.empty:
            # Try partial name match
            qb_weekly = self._weekly_data[
                (self._weekly_data['player_display_name'].str.contains(current_qb.split()[-1], case=False, na=False)) &
                (self._weekly_data['recent_team'] == team_abbrev) &
                (self._weekly_data['season'] == 2025) &
                (self._weekly_data['position'] == 'QB')
            ]

        if qb_weekly.empty:
            return {
                'qb_name': current_qb,
                'team': team_abbrev,
                'games': 0,
                'passing_epa': None,
                'cpoe': None,
                'passing_yards': None,
                'passing_tds': None,
                'interceptions': None,
                'completion_pct': None,
                'confidence': 'none',
                'notes': [f'No 2025 data found for {current_qb}']
            }

        # Calculate stats
        total_attempts = qb_weekly['attempts'].sum()
        total_completions = qb_weekly['completions'].sum()

        stats = {
            'qb_name': current_qb,
            'team': team_abbrev,
            'games': len(qb_weekly),
            'passing_epa': qb_weekly['passing_epa'].mean() if 'passing_epa' in qb_weekly.columns else None,
            'cpoe': qb_weekly['pacr'].mean() if 'pacr' in qb_weekly.columns else None,  # Passer rating as proxy
            'passing_yards': qb_weekly['passing_yards'].sum(),
            'passing_tds': qb_weekly['passing_tds'].sum(),
            'interceptions': qb_weekly['interceptions'].sum(),
            'completion_pct': (total_completions / total_attempts * 100) if total_attempts > 0 else None,
            'yards_per_attempt': qb_weekly['passing_yards'].sum() / total_attempts if total_attempts > 0 else None,
            'confidence': confidence,
            'notes': starter_info['notes'],
            'injury_status': starter_info['status']
        }

        # Add confidence adjustment based on sample size
        if stats['games'] < 3:
            stats['confidence'] = 'low'
            stats['notes'].append(f"Small sample size: only {stats['games']} games")

        return stats

    def get_all_qb_stats(self) -> pd.DataFrame:
        """Get current QB stats for all 32 teams."""
        all_stats = []

        for team in self.detector.ESPN_TEAM_IDS.keys():
            stats = self.get_current_qb_stats(team)
            all_stats.append(stats)
            time.sleep(0.1)  # Rate limiting

        return pd.DataFrame(all_stats)


class ModelIntegration:
    """Integrates starter detection with the NFL prediction model."""

    def __init__(self):
        self.detector = StarterDetector()
        self.stats_adjuster = QBStatsAdjuster(self.detector)

    def get_adjusted_team_factors(self, team_abbrev: str) -> dict:
        """
        Get team factors adjusted for current starter.

        This replaces the static team factors with dynamic ones
        that account for who is actually playing.
        """
        # Get current QB stats
        qb_stats = self.stats_adjuster.get_current_qb_stats(team_abbrev)

        # Get starter info for other positions (future expansion)
        # For now, focus on QB as it's most impactful

        return {
            'team': team_abbrev,
            'qb_factor': {
                'name': qb_stats['qb_name'],
                'epa_per_play': qb_stats.get('passing_epa'),
                'cpoe': qb_stats.get('cpoe'),
                'games_as_starter': qb_stats['games'],
                'injury_status': qb_stats.get('injury_status', 'UNKNOWN'),
                'confidence': qb_stats['confidence']
            },
            'warnings': qb_stats['notes'],
            'data_quality': self._assess_data_quality(qb_stats)
        }

    def _assess_data_quality(self, qb_stats: dict) -> str:
        """Assess overall data quality for model confidence."""
        if qb_stats['confidence'] == 'none':
            return 'UNRELIABLE'
        elif qb_stats['confidence'] == 'low':
            return 'LOW'
        elif qb_stats['games'] < 5:
            return 'MODERATE'
        else:
            return 'HIGH'

    def generate_starter_report(self) -> pd.DataFrame:
        """Generate report of all current starters with confidence levels."""
        starters = self.detector.get_all_starters()

        report_data = []
        for team, info in starters.items():
            report_data.append({
                'team': team,
                'qb1_listed': info['starter'],
                'qb1_status': info['status'],
                'actual_starter': info.get('actual_starter', info['starter']),
                'backup': info['backup'],
                'games_started': info.get('games_started', 0),
                'confidence': info['confidence'],
                'notes': '; '.join(info['notes']) if info['notes'] else ''
            })

        df = pd.DataFrame(report_data)
        return df.sort_values('confidence', ascending=True)

    def get_low_confidence_teams(self) -> List[str]:
        """Get list of teams where starter data is uncertain."""
        starters = self.detector.get_all_starters()

        low_confidence = []
        for team, info in starters.items():
            if info['confidence'] in ['low', 'none']:
                low_confidence.append(team)

        return low_confidence


def refresh_all_data():
    """Refresh all starter and injury data - run before model predictions."""
    print("=" * 60)
    print("NFL STARTER & INJURY DATA REFRESH")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    integration = ModelIntegration()

    # Generate starter report
    print("Fetching depth charts and injury reports...")
    report = integration.generate_starter_report()

    # Save to CSV
    output_file = OUTPUTS_DIR / 'current_starters.csv'
    report.to_csv(output_file, index=False)
    print(f"\nSaved starter report to {output_file}")

    # Show summary
    print("\n" + "=" * 60)
    print("STARTER CONFIDENCE SUMMARY")
    print("=" * 60)

    confidence_counts = report['confidence'].value_counts()
    for conf, count in confidence_counts.items():
        print(f"  {conf.upper()}: {count} teams")

    # Show low confidence teams
    low_conf = report[report['confidence'].isin(['low', 'none'])]
    if not low_conf.empty:
        print("\n  LOW CONFIDENCE TEAMS (review manually):")
        for _, row in low_conf.iterrows():
            print(f"  {row['team']}: {row['actual_starter']} - {row['notes']}")

    # Show injured starters
    injured = report[report['qb1_status'] != 'HEALTHY']
    if not injured.empty:
        print("\n  INJURED STARTING QBs:")
        for _, row in injured.iterrows():
            print(f"  {row['team']}: {row['qb1_listed']} ({row['qb1_status']}) -> {row['actual_starter']}")

    print("\n" + "=" * 60)
    print("DATA REFRESH COMPLETE")
    print("=" * 60)

    return report


# Example usage and CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'refresh':
            refresh_all_data()
        elif sys.argv[1] == 'team' and len(sys.argv) > 2:
            team = sys.argv[2].upper()
            detector = StarterDetector()
            info = detector.get_team_starter(team)
            print(f"\n{team} Starter Info:")
            print(json.dumps(info, indent=2))
        else:
            print("Usage:")
            print("  python starter_detector.py refresh    # Refresh all data")
            print("  python starter_detector.py team ARI   # Get specific team")
    else:
        # Demo run
        print("Running demo...")
        refresh_all_data()
