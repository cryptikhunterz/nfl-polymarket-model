#!/usr/bin/env python3
"""
Generate master_power_ratings.csv from factor files and ESPN API standings.
This combines all 7 factors into weighted power rating models.
"""

import os
import json
import ssl
import urllib.request
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "master_power_ratings.csv"

# Team name mappings
TEAM_NAMES = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LAR': 'Los Angeles Rams', 'LAC': 'Los Angeles Chargers',
    'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks', 'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
}

# ESPN to standard abbreviation mapping
# Using LAR for Rams (consistent with most data sources)
ESPN_ABBR_MAP = {
    'WSH': 'WAS',  # ESPN might use WSH for Commanders
}


def calculate_win_pct_score(df):
    """
    Calculate win percentage score from standings data.

    Win percentage is directly converted to 0-100 scale:
    - 0 wins = 0 score
    - 8.5 wins (0.500) = 50 score
    - 17 wins (1.000) = 100 score
    """
    total_games = df['wins'] + df['losses']
    df['win_pct'] = df['wins'] / total_games.where(total_games > 0, 1)
    df['win_pct_score'] = df['win_pct'] * 100

    # Save factor file
    wins_df = df[['team', 'wins', 'losses', 'win_pct', 'win_pct_score']].copy()
    wins_df.to_csv(PROCESSED_DIR / 'factor_wins.csv', index=False)
    print(f"  Saved factor_wins.csv with {len(wins_df)} teams")

    return df


def fetch_espn_standings():
    """Fetch current standings from ESPN API."""
    url = "https://site.api.espn.com/apis/v2/sports/football/nfl/standings"

    print("Fetching standings from ESPN API...")
    # Create SSL context that doesn't verify certificates (for macOS compatibility)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(url, context=ssl_context) as response:
        data = json.loads(response.read().decode())

    teams = []
    for conf in data.get('children', []):
        for entry in conf.get('standings', {}).get('entries', []):
            team = entry.get('team', {})
            stats = {s['name']: s.get('value', 0) for s in entry.get('stats', [])}

            abbr = team.get('abbreviation', 'UNK')
            # Map ESPN abbreviations to standard
            abbr = ESPN_ABBR_MAP.get(abbr, abbr)

            wins = int(stats.get('wins', 0))
            losses = int(stats.get('losses', 0))

            teams.append({
                'team': abbr,
                'wins': wins,
                'losses': losses
            })

    print(f"  Retrieved {len(teams)} teams from ESPN")
    return pd.DataFrame(teams)


def load_factor_file(filename, score_column):
    """Load a factor file and extract the score column."""
    filepath = PROCESSED_DIR / filename
    if not filepath.exists():
        print(f"  WARNING: {filename} not found")
        return None

    df = pd.read_csv(filepath)
    if score_column not in df.columns:
        print(f"  WARNING: {score_column} not found in {filename}")
        print(f"    Available columns: {list(df.columns)}")
        return None

    # Normalize team abbreviations
    # Using LAR for Rams (consistent with most data sources)
    TEAM_ABBR_NORMALIZE = {
        'LA': 'LAR',   # Rams: normalize to LAR
        'WSH': 'WAS',  # Commanders: some sources use WSH
    }
    df['team'] = df['team'].replace(TEAM_ABBR_NORMALIZE)

    # Return team and score only
    result = df[['team', score_column]].copy()
    print(f"  Loaded {filename}: {len(result)} teams")
    return result


def calculate_power_ratings(df):
    """Calculate various power rating models from factor scores.

    REBALANCED: Wins now weighted at 40% for main models.
    Following Taleb/Silver reasoning - by Week 13, record matters.
    """

    # Main Power (Balanced - wins weighted 40%, process factors 60%)
    # Each factor is already 0-100 normalized
    df['main_power'] = (
        df['win_pct_score'] * 0.40 +        # 40% - wins matter most by Week 13
        df['efficiency_score'] * 0.10 +
        df['oline_score'] * 0.10 +
        df['situational_score'] * 0.10 +
        df['dline_score'] * 0.10 +
        df['ngs_receiving_score'] * 0.10 +
        df['qb_quality_score'] * 0.10
    )

    # QB Heavy Power (wins 35%, QB 25%, others 8% each)
    df['qb_heavy_power'] = (
        df['win_pct_score'] * 0.35 +        # 35% wins
        df['qb_quality_score'] * 0.25 +     # 25% QB emphasis
        df['efficiency_score'] * 0.08 +
        df['oline_score'] * 0.08 +
        df['situational_score'] * 0.08 +
        df['dline_score'] * 0.08 +
        df['ngs_receiving_score'] * 0.08
    )

    # Efficiency Heavy (wins 35%, efficiency 25%, others 8% each)
    df['efficiency_heavy_power'] = (
        df['win_pct_score'] * 0.35 +        # 35% wins
        df['efficiency_score'] * 0.25 +     # 25% efficiency emphasis
        df['qb_quality_score'] * 0.08 +
        df['oline_score'] * 0.08 +
        df['situational_score'] * 0.08 +
        df['dline_score'] * 0.08 +
        df['ngs_receiving_score'] * 0.08
    )

    # Trenches Power (wins 30%, OL 20%, DL 20%, others 7.5% each)
    df['trenches_power'] = (
        df['win_pct_score'] * 0.30 +        # 30% wins
        df['oline_score'] * 0.20 +          # 20% OL emphasis
        df['dline_score'] * 0.20 +          # 20% DL emphasis
        df['qb_quality_score'] * 0.075 +
        df['efficiency_score'] * 0.075 +
        df['situational_score'] * 0.075 +
        df['ngs_receiving_score'] * 0.075
    )

    # Process Power (excludes wins - pure underlying skill factors)
    # UNCHANGED - intentionally excludes wins for comparison
    df['process_power'] = (
        df['efficiency_score'] * 0.20 +
        df['oline_score'] * 0.20 +
        df['dline_score'] * 0.20 +
        df['qb_quality_score'] * 0.20 +
        df['situational_score'] * 0.10 +
        df['ngs_receiving_score'] * 0.10
        # win_pct_score excluded - pure process metrics
    )

    # Skill Only Power (QB + Receiving + Efficiency)
    # UNCHANGED
    df['skill_only_power'] = (
        df['qb_quality_score'] * 0.35 +
        df['ngs_receiving_score'] * 0.35 +
        df['efficiency_score'] * 0.30
    )

    return df


def main():
    print("=" * 60)
    print("GENERATING MASTER POWER RATINGS")
    print("=" * 60)

    # Step 1: Fetch standings
    standings_df = fetch_espn_standings()

    # Step 1b: Calculate win_pct_score from standings
    print("\nCalculating win percentage score...")
    standings_df = calculate_win_pct_score(standings_df)

    # Step 2: Load all factor files
    print("\nLoading factor files...")

    # FIXED: Removed luck_score - replaced with win_pct_score calculated from standings
    factors = {
        'efficiency_score': load_factor_file('factor_efficiency.csv', 'efficiency_score'),
        'oline_score': load_factor_file('factor_line_play.csv', 'line_play_score'),
        'situational_score': load_factor_file('factor_situational.csv', 'situational_score'),
        # luck_score REMOVED - was inverted and rewarding bad teams
        'dline_score': load_factor_file('factor_dline.csv', 'dline_composite_score'),
        'ngs_receiving_score': load_factor_file('factor_ngs_receiving.csv', 'receiving_score'),
        'qb_quality_score': load_factor_file('factor_qb_quality.csv', 'qb_quality_score'),
    }

    # Step 3: Merge all data
    print("\nMerging data...")
    master_df = standings_df.copy()

    for score_name, factor_df in factors.items():
        if factor_df is not None:
            # Rename the score column to our standard name
            orig_col = factor_df.columns[1]  # Second column is the score
            factor_df = factor_df.rename(columns={orig_col: score_name})
            master_df = master_df.merge(factor_df, on='team', how='left')

    # Step 4: Add team names
    master_df['name'] = master_df['team'].map(TEAM_NAMES)

    # Step 5: Fill missing values with league average (50)
    score_cols = [c for c in master_df.columns if c.endswith('_score')]
    for col in score_cols:
        missing = master_df[col].isna().sum()
        if missing > 0:
            print(f"  WARNING: {missing} teams missing {col}, filling with 50")
        master_df[col] = master_df[col].fillna(50)

    # Step 6: Calculate power ratings
    print("\nCalculating power ratings...")
    master_df = calculate_power_ratings(master_df)

    # Step 7: Reorder columns
    # FIXED: Using win_pct_score instead of luck_score, process_power instead of anti_luck_power
    column_order = [
        'team', 'name', 'wins', 'losses', 'win_pct', 'win_pct_score',
        'efficiency_score', 'oline_score', 'situational_score',
        'dline_score', 'ngs_receiving_score', 'qb_quality_score',
        'main_power', 'qb_heavy_power', 'efficiency_heavy_power',
        'trenches_power', 'process_power', 'skill_only_power'
    ]
    master_df = master_df[column_order]

    # Step 8: Save
    master_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")

    # Step 9: Show top 10
    print("\n" + "=" * 60)
    print("TOP 10 TEAMS BY MAIN POWER RATING")
    print("=" * 60)

    top10 = master_df.nlargest(10, 'main_power')[['team', 'name', 'wins', 'losses', 'main_power']]
    for i, row in enumerate(top10.itertuples(), 1):
        print(f"{i:2}. {row.team} ({row.wins}-{row.losses}): {row.main_power:.2f}")

    print("\n" + "=" * 60)
    print("BOTTOM 5 TEAMS BY MAIN POWER RATING")
    print("=" * 60)

    bottom5 = master_df.nsmallest(5, 'main_power')[['team', 'name', 'wins', 'losses', 'main_power']]
    for i, row in enumerate(bottom5.itertuples(), 1):
        print(f"{27+i:2}. {row.team} ({row.wins}-{row.losses}): {row.main_power:.2f}")  # Fixed: 27+i for ranks 28-32

    return master_df


if __name__ == "__main__":
    main()
