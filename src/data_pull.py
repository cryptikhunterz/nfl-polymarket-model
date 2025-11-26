"""
NFL Data Pull Module
Fetches data from nfl_data_py, FiveThirtyEight, and other sources
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import requests
from pathlib import Path
from datetime import datetime
import ssl
import certifi

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Data directories
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def pull_play_by_play(years: list = None) -> pd.DataFrame:
    """
    Pull play-by-play data for specified years.
    Default: 2023 and 2024 seasons (1.5-2 season lookback)
    """
    if years is None:
        years = [2023, 2024]

    print(f"Pulling play-by-play data for {years}...")
    try:
        pbp = nfl.import_pbp_data(years)

        if pbp is None or len(pbp) == 0 or 'season' not in pbp.columns:
            print("  Warning: PBP data returned empty. Using cached data if available.")
            # Try to load cached data
            cached_dfs = []
            for year in years:
                cached_file = RAW_DIR / f"pbp_{year}.csv"
                if cached_file.exists():
                    cached_dfs.append(pd.read_csv(cached_file))
                    print(f"    Loaded cached pbp_{year}.csv")
            if cached_dfs:
                return pd.concat(cached_dfs, ignore_index=True)
            return None

        # Save to raw directory
        for year in years:
            year_data = pbp[pbp['season'] == year]
            if len(year_data) > 0:
                year_data.to_csv(RAW_DIR / f"pbp_{year}.csv", index=False)
                print(f"  Saved pbp_{year}.csv ({len(year_data)} plays)")

        return pbp
    except Exception as e:
        print(f"  Error pulling PBP data: {e}")
        # Try to load cached data
        cached_dfs = []
        for year in years:
            cached_file = RAW_DIR / f"pbp_{year}.csv"
            if cached_file.exists():
                cached_dfs.append(pd.read_csv(cached_file))
                print(f"    Loaded cached pbp_{year}.csv")
        if cached_dfs:
            return pd.concat(cached_dfs, ignore_index=True)
        return None


def pull_weekly_data(years: list = None) -> pd.DataFrame:
    """Pull weekly team/player stats"""
    if years is None:
        years = [2023, 2024]

    print(f"Pulling weekly data for {years}...")
    weekly = nfl.import_weekly_data(years)
    weekly.to_csv(RAW_DIR / "weekly_stats.csv", index=False)
    print(f"  Saved weekly_stats.csv ({len(weekly)} rows)")

    return weekly


def pull_seasonal_data(years: list = None) -> pd.DataFrame:
    """Pull seasonal stats"""
    if years is None:
        years = [2023, 2024]

    print(f"Pulling seasonal data for {years}...")
    seasonal = nfl.import_seasonal_data(years)
    seasonal.to_csv(RAW_DIR / "seasonal_stats.csv", index=False)
    print(f"  Saved seasonal_stats.csv ({len(seasonal)} rows)")

    return seasonal


def pull_team_descriptions() -> pd.DataFrame:
    """Pull team info (names, abbreviations, etc.)"""
    print("Pulling team descriptions...")
    teams = nfl.import_team_desc()
    teams.to_csv(RAW_DIR / "teams.csv", index=False)
    print(f"  Saved teams.csv ({len(teams)} teams)")

    return teams


def pull_schedules(years: list = None) -> pd.DataFrame:
    """Pull game schedules and results"""
    if years is None:
        years = [2023, 2024]

    print(f"Pulling schedules for {years}...")
    schedules = nfl.import_schedules(years)
    schedules.to_csv(RAW_DIR / "schedules.csv", index=False)
    print(f"  Saved schedules.csv ({len(schedules)} games)")

    return schedules


def pull_fivethirtyeight_elo() -> pd.DataFrame:
    """
    Download FiveThirtyEight NFL Elo ratings
    Source: https://github.com/fivethirtyeight/data/tree/master/nfl-elo
    """
    print("Downloading FiveThirtyEight Elo data...")

    url = "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv"

    try:
        elo_df = pd.read_csv(url)

        # Filter to recent seasons
        elo_df['date'] = pd.to_datetime(elo_df['date'])
        recent_elo = elo_df[elo_df['season'] >= 2023]

        recent_elo.to_csv(RAW_DIR / "elo_538.csv", index=False)
        print(f"  Saved elo_538.csv ({len(recent_elo)} games)")

        return recent_elo

    except Exception as e:
        print(f"  Error downloading 538 Elo: {e}")
        print("  Trying alternative source...")

        # Alternative: GitHub raw file
        alt_url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo.csv"
        try:
            elo_df = pd.read_csv(alt_url)
            elo_df['date'] = pd.to_datetime(elo_df['date'])
            recent_elo = elo_df[elo_df['season'] >= 2023]
            recent_elo.to_csv(RAW_DIR / "elo_538.csv", index=False)
            print(f"  Saved elo_538.csv ({len(recent_elo)} games)")
            return recent_elo
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")
            return None


def pull_rosters(years: list = None) -> pd.DataFrame:
    """Pull roster data for player info"""
    if years is None:
        years = [2023, 2024]

    print(f"Pulling rosters for {years}...")
    try:
        # Try the private method name (API changed)
        if hasattr(nfl, 'import_rosters'):
            rosters = nfl.import_rosters(years)
        else:
            rosters = nfl._import_rosters(years)
        rosters.to_csv(RAW_DIR / "rosters.csv", index=False)
        print(f"  Saved rosters.csv ({len(rosters)} players)")
        return rosters
    except Exception as e:
        print(f"  Rosters not available: {e}")
        return None


def pull_qbr_data(years: list = None) -> pd.DataFrame:
    """Pull ESPN QBR data if available"""
    if years is None:
        years = [2023, 2024]

    print(f"Pulling QBR data for {years}...")
    try:
        qbr = nfl.import_qbr(years)
        qbr.to_csv(RAW_DIR / "qbr.csv", index=False)
        print(f"  Saved qbr.csv ({len(qbr)} rows)")
        return qbr
    except Exception as e:
        print(f"  QBR data not available: {e}")
        return None


def calculate_team_epa_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team-level EPA stats from play-by-play data.
    This is a key input for the factor calculations.
    """
    print("Calculating team EPA stats...")

    # Filter to real plays (not penalties, timeouts, etc.)
    plays = pbp[
        (pbp['play_type'].isin(['pass', 'run'])) &
        (pbp['epa'].notna())
    ].copy()

    # Offensive EPA per play
    off_epa = plays.groupby(['season', 'week', 'posteam']).agg({
        'epa': ['mean', 'sum', 'count'],
        'success': 'mean',
        'yards_gained': 'mean'
    }).reset_index()
    off_epa.columns = ['season', 'week', 'team', 'off_epa_play', 'off_epa_total',
                       'off_plays', 'success_rate', 'yards_per_play']

    # Defensive EPA per play (EPA allowed)
    def_epa = plays.groupby(['season', 'week', 'defteam']).agg({
        'epa': ['mean', 'sum', 'count'],
        'success': 'mean'
    }).reset_index()
    def_epa.columns = ['season', 'week', 'team', 'def_epa_play', 'def_epa_total',
                       'def_plays', 'def_success_rate']

    # Merge offensive and defensive stats
    team_epa = pd.merge(off_epa, def_epa, on=['season', 'week', 'team'], how='outer')

    # Save
    team_epa.to_csv(PROCESSED_DIR / "team_epa_weekly.csv", index=False)
    print(f"  Saved team_epa_weekly.csv ({len(team_epa)} team-weeks)")

    return team_epa


def calculate_qb_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate QB-level stats for QB Quality factor.
    EPA/dropback, CPOE, pressure-to-sack rate, etc.
    """
    print("Calculating QB stats...")

    # Filter to pass plays
    passes = pbp[
        (pbp['play_type'] == 'pass') &
        (pbp['passer_player_name'].notna())
    ].copy()

    # Group by QB
    qb_stats = passes.groupby(['season', 'week', 'passer_player_name', 'posteam']).agg({
        'epa': ['mean', 'sum'],
        'cpoe': 'mean',
        'complete_pass': 'mean',
        'interception': 'sum',
        'sack': 'sum',
        'play_id': 'count',
        'air_yards': 'mean',
        'yards_after_catch': 'mean',
        'pass_touchdown': 'sum'
    }).reset_index()

    qb_stats.columns = ['season', 'week', 'qb_name', 'team', 'epa_dropback', 'epa_total',
                        'cpoe', 'completion_pct', 'interceptions', 'sacks', 'dropbacks',
                        'avg_air_yards', 'avg_yac', 'pass_tds']

    # Save
    qb_stats.to_csv(PROCESSED_DIR / "qb_stats_weekly.csv", index=False)
    print(f"  Saved qb_stats_weekly.csv ({len(qb_stats)} QB-weeks)")

    return qb_stats


def pull_all_data():
    """Main function to pull all required data"""
    print("=" * 60)
    print("NFL DATA PULL")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    years = [2023, 2024]

    # Pull raw data
    pbp = pull_play_by_play(years)
    pull_weekly_data(years)
    pull_seasonal_data(years)
    pull_team_descriptions()
    pull_schedules(years)
    pull_fivethirtyeight_elo()
    pull_rosters(years)
    pull_qbr_data(years)

    # Calculate derived stats
    if pbp is not None:
        calculate_team_epa_stats(pbp)
        calculate_qb_stats(pbp)

    print("\n" + "=" * 60)
    print("DATA PULL COMPLETE")
    print("=" * 60)

    # List files created
    print("\nRaw data files:")
    for f in RAW_DIR.glob("*.csv"):
        print(f"  {f.name}")

    print("\nProcessed data files:")
    for f in PROCESSED_DIR.glob("*.csv"):
        print(f"  {f.name}")


if __name__ == "__main__":
    pull_all_data()
