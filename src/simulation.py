"""
NFL Monte Carlo Simulation
Simulates 10,000 playoff brackets to estimate Super Bowl win probability
"""

import pandas as pd
import numpy as np
from scipy.stats import beta, norm
from pathlib import Path
from typing import Dict, List, Tuple
import random

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# NFL Playoff structure (2024 format)
# AFC/NFC each have:
# - 7 teams (4 division winners + 3 wild cards)
# - #1 seed gets bye, #2-7 play wild card round
# - Wild Card: #2 vs #7, #3 vs #6, #4 vs #5
# - Divisional: #1 vs lowest remaining, higher vs lower
# - Conference Championship
# - Super Bowl


class PlayoffSimulator:
    """Monte Carlo simulation of NFL playoffs and Super Bowl"""

    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        self.team_probs = None
        self.elo_ratings = None

    def load_data(self):
        """Load team probabilities and Elo ratings"""
        # Load championship probabilities from model
        prob_file = OUTPUTS_DIR / "championship_probabilities.csv"
        if prob_file.exists():
            self.team_probs = pd.read_csv(prob_file)
            print(f"Loaded probabilities for {len(self.team_probs)} teams")

        # Load Elo ratings
        elo_file = PROCESSED_DIR / "unit_elos.csv"
        if elo_file.exists():
            self.elo_ratings = pd.read_csv(elo_file)
            print(f"Loaded Elo ratings")

    def get_team_strength(self, team: str) -> float:
        """Get team strength score for head-to-head simulation"""
        if self.team_probs is not None:
            team_row = self.team_probs[self.team_probs['team'] == team]
            if len(team_row) > 0:
                return team_row.iloc[0]['score_ensemble']

        if self.elo_ratings is not None:
            elo_row = self.elo_ratings[self.elo_ratings['team'] == team]
            if len(elo_row) > 0:
                return elo_row.iloc[0]['team_elo']

        return 1500  # Default Elo

    def simulate_game(self, home_team: str, away_team: str,
                      home_field_advantage: float = 3.0) -> str:
        """
        Simulate a single game between two teams.
        Returns the winning team.

        Uses team strength with home field advantage and randomness.
        """
        home_strength = self.get_team_strength(home_team) + home_field_advantage
        away_strength = self.get_team_strength(away_team)

        # Convert to win probability using logistic function
        strength_diff = home_strength - away_strength

        # For Elo: diff of 100 = ~64% win probability
        # For score: diff of 10 = ~64% win probability
        if home_strength > 100:  # Likely Elo scale
            scale = 400
        else:  # Score scale (0-100)
            scale = 40

        home_win_prob = 1 / (1 + 10 ** (-strength_diff / scale))

        # Add some randomness via Beta distribution
        # This models uncertainty in any given game
        alpha = home_win_prob * 10
        beta_param = (1 - home_win_prob) * 10
        actual_prob = beta.rvs(alpha, beta_param)

        # Simulate game
        if random.random() < actual_prob:
            return home_team
        else:
            return away_team

    def simulate_playoff_round(self, matchups: List[Tuple[str, str]],
                               higher_seed_home: bool = True) -> List[str]:
        """
        Simulate a round of playoff games.
        Returns list of winners.

        matchups: List of (home_team, away_team) or (higher_seed, lower_seed)
        """
        winners = []
        for home, away in matchups:
            # In playoffs, higher seed hosts
            hfa = 3.0 if higher_seed_home else 0.0
            winner = self.simulate_game(home, away, hfa)
            winners.append(winner)
        return winners

    def simulate_conference_playoff(self, seeded_teams: List[str]) -> str:
        """
        Simulate one conference's playoff bracket.

        seeded_teams: List of 7 teams in seed order [1-7]
        Returns: Conference champion
        """
        if len(seeded_teams) != 7:
            raise ValueError("Need exactly 7 teams for conference playoff")

        seeds = {team: i+1 for i, team in enumerate(seeded_teams)}

        # Wild Card Round (#1 has bye)
        # #2 vs #7, #3 vs #6, #4 vs #5
        wc_matchups = [
            (seeded_teams[1], seeded_teams[6]),  # #2 vs #7
            (seeded_teams[2], seeded_teams[5]),  # #3 vs #6
            (seeded_teams[3], seeded_teams[4]),  # #4 vs #5
        ]
        wc_winners = self.simulate_playoff_round(wc_matchups)

        # Divisional Round
        # #1 vs lowest remaining, other two play each other
        remaining = [seeded_teams[0]] + wc_winners  # #1 + 3 WC winners
        remaining_seeds = [(t, seeds[t]) for t in remaining]
        remaining_seeds.sort(key=lambda x: x[1])  # Sort by seed

        # #1 plays lowest seed, #2/#3 seeds play each other
        div_matchups = [
            (remaining_seeds[0][0], remaining_seeds[3][0]),  # #1 vs lowest
            (remaining_seeds[1][0], remaining_seeds[2][0]),  # middle seeds
        ]
        div_winners = self.simulate_playoff_round(div_matchups)

        # Conference Championship
        div_winner_seeds = [(t, seeds[t]) for t in div_winners]
        div_winner_seeds.sort(key=lambda x: x[1])

        conf_matchup = [(div_winner_seeds[0][0], div_winner_seeds[1][0])]
        conf_winner = self.simulate_playoff_round(conf_matchup)[0]

        return conf_winner

    def simulate_super_bowl(self, afc_champ: str, nfc_champ: str) -> str:
        """Simulate Super Bowl (neutral site, no HFA)"""
        return self.simulate_game(afc_champ, nfc_champ, home_field_advantage=0)

    def get_playoff_teams(self) -> Dict[str, List[str]]:
        """
        Get current playoff teams or project based on standings.
        Returns dict with 'AFC' and 'NFC' keys, each with 7 teams in seed order.
        """
        # For now, use model predictions to create projected playoff field
        # In production, this would read from actual standings

        if self.team_probs is None:
            return {'AFC': [], 'NFC': []}

        # NFL team conference mapping
        afc_teams = ['BUF', 'MIA', 'NE', 'NYJ',  # AFC East
                     'BAL', 'CIN', 'CLE', 'PIT',  # AFC North
                     'HOU', 'IND', 'JAX', 'TEN',  # AFC South
                     'DEN', 'KC', 'LV', 'LAC']    # AFC West

        nfc_teams = ['DAL', 'NYG', 'PHI', 'WAS',  # NFC East
                     'CHI', 'DET', 'GB', 'MIN',   # NFC North
                     'ATL', 'CAR', 'NO', 'TB',    # NFC South
                     'ARI', 'LAR', 'SF', 'SEA']   # NFC West

        # Get top 7 from each conference by model score
        afc_probs = self.team_probs[self.team_probs['team'].isin(afc_teams)]
        nfc_probs = self.team_probs[self.team_probs['team'].isin(nfc_teams)]

        afc_playoff = afc_probs.nlargest(7, 'score_ensemble')['team'].tolist()
        nfc_playoff = nfc_probs.nlargest(7, 'score_ensemble')['team'].tolist()

        return {'AFC': afc_playoff, 'NFC': nfc_playoff}

    def run_simulation(self) -> pd.DataFrame:
        """
        Run full Monte Carlo simulation.
        Returns DataFrame with Super Bowl win counts per team.
        """
        print(f"\nRunning {self.n_simulations:,} playoff simulations...")

        self.load_data()

        if self.team_probs is None:
            print("Error: No team probabilities found. Run model.py first.")
            return None

        playoff_teams = self.get_playoff_teams()

        if len(playoff_teams['AFC']) < 7 or len(playoff_teams['NFC']) < 7:
            print("Error: Not enough teams for playoff simulation")
            return None

        # Track results
        sb_wins = {}
        conf_champ_appearances = {}
        sb_appearances = {}

        for team in playoff_teams['AFC'] + playoff_teams['NFC']:
            sb_wins[team] = 0
            conf_champ_appearances[team] = 0
            sb_appearances[team] = 0

        # Run simulations
        for i in range(self.n_simulations):
            if (i + 1) % 2000 == 0:
                print(f"  Completed {i+1:,} simulations...")

            # Simulate each conference
            afc_champ = self.simulate_conference_playoff(playoff_teams['AFC'])
            nfc_champ = self.simulate_conference_playoff(playoff_teams['NFC'])

            conf_champ_appearances[afc_champ] += 1
            conf_champ_appearances[nfc_champ] += 1

            sb_appearances[afc_champ] += 1
            sb_appearances[nfc_champ] += 1

            # Simulate Super Bowl
            sb_winner = self.simulate_super_bowl(afc_champ, nfc_champ)
            sb_wins[sb_winner] += 1

        # Compile results
        results = []
        for team in sb_wins.keys():
            results.append({
                'team': team,
                'sb_wins': sb_wins[team],
                'sb_win_pct': sb_wins[team] / self.n_simulations * 100,
                'sb_appearances': sb_appearances[team],
                'sb_appearance_pct': sb_appearances[team] / self.n_simulations * 100,
                'conf_champ_appearances': conf_champ_appearances[team],
                'conf_champ_pct': conf_champ_appearances[team] / self.n_simulations * 100
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sb_win_pct', ascending=False).reset_index(drop=True)

        # Add model probability for comparison
        if self.team_probs is not None:
            results_df = results_df.merge(
                self.team_probs[['team', 'championship_pct']],
                on='team',
                how='left'
            )
            results_df.rename(columns={'championship_pct': 'model_prob_pct'}, inplace=True)

        # Save results
        results_df.to_csv(OUTPUTS_DIR / "simulation_results.csv", index=False)

        print("\n" + "=" * 60)
        print(f"MONTE CARLO RESULTS ({self.n_simulations:,} simulations)")
        print("=" * 60)
        print(results_df[['team', 'sb_win_pct', 'sb_appearance_pct', 'model_prob_pct']].head(14).to_string(index=False))

        return results_df


def run_monte_carlo():
    """Main function to run Monte Carlo simulation"""
    simulator = PlayoffSimulator(n_simulations=10000)
    results = simulator.run_simulation()
    return results


if __name__ == "__main__":
    run_monte_carlo()
