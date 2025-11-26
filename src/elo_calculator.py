"""
NFL Elo Calculator Module
FiveThirtyEight exact implementation + Unit-specific Elos

Team Elo: FiveThirtyEight methodology
- Source: https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py
- K=20, Starting=1505, Home advantage=48, Season regression=1/3 toward 1505

Unit Elo: Custom implementation for Offense, Defense, Special Teams
"""

import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


class TeamElo:
    """
    FiveThirtyEight NFL Elo implementation
    Verified against their GitHub: https://github.com/fivethirtyeight/nfl-elo-game
    """

    def __init__(self):
        self.K = 20                    # How much one game shifts ratings
        self.HOME_ADV = 48             # Add to home team before calculating expected
        self.START_ELO = 1505          # Slightly above 1500 for expansion teams
        self.PLAYOFF_MULT = 1.2        # Multiply elo diff for playoff win probability
        self.ratings: Dict[str, float] = {}  # {team_name: elo}
        self.history = []              # Track all game updates

    def initialize_team(self, team: str, elo: Optional[float] = None):
        """Set initial Elo for a team"""
        if elo is None:
            elo = self.START_ELO
        self.ratings[team] = elo

    def get_elo(self, team: str) -> float:
        """Get team's current Elo, initialize if new"""
        if team not in self.ratings:
            self.initialize_team(team)
        return self.ratings[team]

    def expected_result(self, team_elo: float, opponent_elo: float) -> float:
        """
        Win probability based on Elo difference.
        Formula: 1 / (1 + 10^((Opponent_Elo - Team_Elo) / 400))
        """
        return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))

    def expected_result_with_home(self, team_elo: float, opponent_elo: float,
                                   team_is_home: bool) -> float:
        """Win probability including home field advantage (+48 to home team)"""
        if team_is_home:
            adj_team = team_elo + self.HOME_ADV
            adj_opp = opponent_elo
        else:
            adj_team = team_elo
            adj_opp = opponent_elo + self.HOME_ADV

        return self.expected_result(adj_team, adj_opp)

    def mov_multiplier(self, point_diff: int, winner_elo: float, loser_elo: float) -> float:
        """
        Margin of victory multiplier - EXACT 538 formula.

        Formula: ln(|Point_Diff| + 1) × (2.2 / ((Winner_Elo - Loser_Elo) × 0.001 + 2.2))

        CRITICAL: Winner_Elo - Loser_Elo is SIGNED, not absolute value.
        - If favorite wins: positive elo_diff → larger denominator → smaller multiplier
        - If underdog wins: negative elo_diff → smaller denominator → larger multiplier
        """
        elo_diff = winner_elo - loser_elo  # SIGNED difference
        return math.log(max(abs(point_diff), 1) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

    def update_after_game(self, team: str, opponent: str,
                          team_score: int, opponent_score: int,
                          team_is_home: bool,
                          season: int = None, week: int = None) -> Tuple[float, float]:
        """
        Update both teams' Elo after a game.

        Returns: (new_team_elo, new_opponent_elo)
        """
        team_elo = self.get_elo(team)
        opp_elo = self.get_elo(opponent)

        pre_team_elo = team_elo
        pre_opp_elo = opp_elo

        # Determine winner
        point_diff = abs(team_score - opponent_score)

        # Expected result (with home adjustment)
        team_expected = self.expected_result_with_home(team_elo, opp_elo, team_is_home)
        opp_expected = 1 - team_expected

        # Actual result and MOV
        if team_score > opponent_score:
            team_actual = 1
            opp_actual = 0
            winner_elo = team_elo
            loser_elo = opp_elo
        elif team_score < opponent_score:
            team_actual = 0
            opp_actual = 1
            winner_elo = opp_elo
            loser_elo = team_elo
        else:  # Tie (rare in NFL)
            team_actual = 0.5
            opp_actual = 0.5
            winner_elo = team_elo  # Doesn't matter for tie
            loser_elo = opp_elo

        # MOV multiplier
        if point_diff == 0:  # Tie
            mov = 1.525  # 538 uses 2.2 * ln(2) for ties ≈ 1.525
        else:
            mov = self.mov_multiplier(point_diff, winner_elo, loser_elo)

        # Update ratings: New_Elo = Old_Elo + K × MOV × (Actual - Expected)
        team_shift = self.K * mov * (team_actual - team_expected)
        opp_shift = self.K * mov * (opp_actual - opp_expected)

        self.ratings[team] = team_elo + team_shift
        self.ratings[opponent] = opp_elo + opp_shift

        # Record history
        self.history.append({
            'season': season,
            'week': week,
            'home_team': team if team_is_home else opponent,
            'away_team': opponent if team_is_home else team,
            'home_score': team_score if team_is_home else opponent_score,
            'away_score': opponent_score if team_is_home else team_score,
            'pre_home_elo': pre_team_elo if team_is_home else pre_opp_elo,
            'pre_away_elo': pre_opp_elo if team_is_home else pre_team_elo,
            'post_home_elo': self.ratings[team] if team_is_home else self.ratings[opponent],
            'post_away_elo': self.ratings[opponent] if team_is_home else self.ratings[team],
            'mov_multiplier': mov,
            'home_expected': team_expected if team_is_home else opp_expected
        })

        return self.ratings[team], self.ratings[opponent]

    def new_season(self):
        """
        Regress all teams 1/3 toward mean between seasons.
        Formula: New_Elo = Old_Elo + (1/3) × (1505 - Old_Elo)
        """
        for team in self.ratings:
            self.ratings[team] = self.ratings[team] + (1/3) * (self.START_ELO - self.ratings[team])

    def elo_to_spread(self, team_elo: float, opponent_elo: float, team_is_home: bool) -> float:
        """
        Convert Elo to point spread.
        100 Elo points = 4 point spread (divide by 25)
        """
        if team_is_home:
            elo_diff = (team_elo + self.HOME_ADV) - opponent_elo
        else:
            elo_diff = team_elo - (opponent_elo + self.HOME_ADV)
        return elo_diff / 25

    def playoff_win_prob(self, team_elo: float, opponent_elo: float, team_is_home: bool) -> float:
        """
        Win probability for playoff games (1.2x multiplier on elo diff).
        Favorites overperform in playoffs.
        """
        if team_is_home:
            elo_diff = (team_elo + self.HOME_ADV) - opponent_elo
        else:
            elo_diff = team_elo - (opponent_elo + self.HOME_ADV)

        # Playoff adjustment
        elo_diff = elo_diff * self.PLAYOFF_MULT

        return 1 / (1 + 10 ** (-elo_diff / 400))

    def get_rankings(self) -> pd.DataFrame:
        """Get current rankings as DataFrame"""
        data = [{'team': t, 'team_elo': e} for t, e in self.ratings.items()]
        df = pd.DataFrame(data)
        df = df.sort_values('team_elo', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        return df


class UnitElo:
    """
    Separate Elo ratings for Offense, Defense, Special Teams.
    Custom implementation for our model.
    """

    def __init__(self):
        self.K = 15                    # Lower K than team Elo — units are noisier
        self.START_ELO = 1505
        self.LEAGUE_AVG_POINTS = 22    # Approximate league average points per game

        # {team: {'off': elo, 'def': elo, 'st': elo}}
        self.ratings: Dict[str, Dict[str, float]] = {}

    def initialize_team(self, team: str):
        """Initialize all three units at starting Elo"""
        if team not in self.ratings:
            self.ratings[team] = {
                'off': self.START_ELO,
                'def': self.START_ELO,
                'st': self.START_ELO
            }

    def get_ratings(self, team: str) -> Dict[str, float]:
        """Get all unit ratings for a team"""
        self.initialize_team(team)
        return self.ratings[team]

    def expected_points(self, off_elo: float, def_elo: float) -> float:
        """
        Expected points when offense with off_elo faces defense with def_elo.
        100 Elo point advantage = 4 more points expected.
        """
        elo_diff = off_elo - def_elo
        return self.LEAGUE_AVG_POINTS + (elo_diff / 25)

    def update_after_game(self, team: str, opponent: str,
                          team_score: int, opponent_score: int) -> Dict:
        """
        Update offensive and defensive Elo for both teams.

        Team's offense vs Opponent's defense: team_score
        Opponent's offense vs Team's defense: opponent_score
        """
        self.initialize_team(team)
        self.initialize_team(opponent)

        # Team's offense vs Opponent's defense
        team_off_elo = self.ratings[team]['off']
        opp_def_elo = self.ratings[opponent]['def']

        expected_team_score = self.expected_points(team_off_elo, opp_def_elo)
        team_score_diff = team_score - expected_team_score

        # Opponent's offense vs Team's defense
        opp_off_elo = self.ratings[opponent]['off']
        team_def_elo = self.ratings[team]['def']

        expected_opp_score = self.expected_points(opp_off_elo, team_def_elo)
        opp_score_diff = opponent_score - expected_opp_score

        # Update offensive ratings
        # Beat expectation = go up, miss expectation = go down
        team_off_shift = self.K * (team_score_diff / 10)  # Scale: 10 points diff = K shift
        opp_off_shift = self.K * (opp_score_diff / 10)

        self.ratings[team]['off'] += team_off_shift
        self.ratings[opponent]['off'] += opp_off_shift

        # Update defensive ratings
        # INVERSE: opponent scoring less than expected = your defense is good
        team_def_shift = self.K * (-opp_score_diff / 10)  # Negative because lower is better
        opp_def_shift = self.K * (-team_score_diff / 10)

        self.ratings[team]['def'] += team_def_shift
        self.ratings[opponent]['def'] += opp_def_shift

        return {
            'team': self.ratings[team].copy(),
            'opponent': self.ratings[opponent].copy()
        }

    def update_special_teams(self, team: str, opponent: str,
                              team_st_points: float, opp_st_points: float):
        """
        Update special teams Elo.

        ST points = field goals made + return TDs + points from field position
        """
        self.initialize_team(team)
        self.initialize_team(opponent)

        team_st_elo = self.ratings[team]['st']
        opp_st_elo = self.ratings[opponent]['st']

        # Expected ST points based on Elo
        expected_team_st = 3 + (team_st_elo - opp_st_elo) / 100  # ~3 ST points average
        expected_opp_st = 3 + (opp_st_elo - team_st_elo) / 100

        # Update based on difference
        team_st_shift = self.K * (team_st_points - expected_team_st) / 3
        opp_st_shift = self.K * (opp_st_points - expected_opp_st) / 3

        self.ratings[team]['st'] += team_st_shift
        self.ratings[opponent]['st'] += opp_st_shift

    def new_season(self):
        """Regress all units 1/3 toward mean"""
        for team in self.ratings:
            for unit in ['off', 'def', 'st']:
                old = self.ratings[team][unit]
                self.ratings[team][unit] = old + (1/3) * (self.START_ELO - old)

    def combined_elo(self, team: str, off_weight: float = 0.4,
                     def_weight: float = 0.4, st_weight: float = 0.2) -> float:
        """
        Combine unit Elos into single team rating.
        Default weights: 40% offense, 40% defense, 20% special teams
        """
        self.initialize_team(team)
        r = self.ratings[team]
        return r['off'] * off_weight + r['def'] * def_weight + r['st'] * st_weight

    def get_rankings(self) -> pd.DataFrame:
        """Get current unit ratings as DataFrame"""
        data = []
        for team, ratings in self.ratings.items():
            data.append({
                'team': team,
                'offense_elo': ratings['off'],
                'defense_elo': ratings['def'],
                'special_teams_elo': ratings['st'],
                'combined_elo': self.combined_elo(team)
            })
        df = pd.DataFrame(data)
        df = df.sort_values('combined_elo', ascending=False).reset_index(drop=True)
        return df


def calculate_elos_from_games(games_df: pd.DataFrame) -> Tuple[TeamElo, UnitElo]:
    """
    Process all games chronologically to calculate Elos.

    Args:
        games_df: DataFrame with columns [season, week, home_team, away_team, home_score, away_score]

    Returns:
        team_elo: TeamElo object with final ratings
        unit_elo: UnitElo object with final ratings
    """
    team_elo = TeamElo()
    unit_elo = UnitElo()

    # Sort by season, then week
    games_df = games_df.sort_values(['season', 'week']).copy()

    current_season = None

    for _, game in games_df.iterrows():
        # Handle season transition
        if current_season is not None and game['season'] != current_season:
            print(f"  Season {current_season} → {game['season']}: Applying regression")
            team_elo.new_season()
            unit_elo.new_season()

        current_season = game['season']

        # Skip games not yet played
        if pd.isna(game['home_score']) or pd.isna(game['away_score']):
            continue

        home = game['home_team']
        away = game['away_team']
        home_score = int(game['home_score'])
        away_score = int(game['away_score'])

        # Update team Elo
        team_elo.update_after_game(
            team=home,
            opponent=away,
            team_score=home_score,
            opponent_score=away_score,
            team_is_home=True,
            season=game['season'],
            week=game['week']
        )

        # Update unit Elo
        unit_elo.update_after_game(home, away, home_score, away_score)

    return team_elo, unit_elo


def calculate_all_elos():
    """Main function to calculate all Elo ratings"""
    print("=" * 60)
    print("ELO CALCULATION (FiveThirtyEight Methodology)")
    print("=" * 60)

    # Load schedule data
    schedule_file = RAW_DIR / "schedules.csv"
    if not schedule_file.exists():
        print("Error: schedules.csv not found. Run data_pull.py first.")
        return None, None

    games_df = pd.read_csv(schedule_file)
    print(f"Loaded {len(games_df)} games from schedules.csv")

    # Process all games
    team_elo, unit_elo = calculate_elos_from_games(games_df)

    # Get completed games count
    completed = games_df[games_df['home_score'].notna()]
    print(f"Processed {len(completed)} completed games")

    # Get Team Elo rankings
    team_rankings = team_elo.get_rankings()
    team_rankings.to_csv(PROCESSED_DIR / "team_elo.csv", index=False)
    print(f"\nSaved team_elo.csv")

    # Get Unit Elo rankings
    unit_rankings = unit_elo.get_rankings()
    unit_rankings.to_csv(PROCESSED_DIR / "unit_elos.csv", index=False)
    print(f"Saved unit_elos.csv")

    # Save Elo history
    if team_elo.history:
        history_df = pd.DataFrame(team_elo.history)
        history_df.to_csv(PROCESSED_DIR / "elo_history.csv", index=False)
        print(f"Saved elo_history.csv")

    # Print Team Elo rankings
    print("\n" + "=" * 60)
    print("TEAM ELO RANKINGS (FiveThirtyEight Method)")
    print("=" * 60)
    print(team_rankings[['rank', 'team', 'team_elo']].head(15).to_string(index=False))

    # Print Unit Elo rankings
    print("\n" + "=" * 60)
    print("UNIT ELO RANKINGS (Top 10 Combined)")
    print("=" * 60)
    print(unit_rankings[['team', 'offense_elo', 'defense_elo', 'special_teams_elo', 'combined_elo']].head(10).to_string(index=False))

    return team_elo, unit_elo


if __name__ == "__main__":
    calculate_all_elos()
