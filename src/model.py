"""
NFL Scoring Model
Implements 5 weight profiles for team scoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# Weight profiles per spec
WEIGHT_PROFILES = {
    'A_baseline': {
        'name': 'Baseline',
        'qb_quality': 0.30,
        'efficiency': 0.25,
        'line_play': 0.20,
        'situational': 0.15,
        'luck_regression': 0.10
    },
    'B_qb_heavy': {
        'name': 'QB Heavy',
        'qb_quality': 0.45,
        'efficiency': 0.20,
        'line_play': 0.15,
        'situational': 0.12,
        'luck_regression': 0.08
    },
    'C_line_heavy': {
        'name': 'Line Heavy',
        'qb_quality': 0.25,
        'efficiency': 0.20,
        'line_play': 0.30,
        'situational': 0.15,
        'luck_regression': 0.10
    },
    'D_efficiency_pure': {
        'name': 'Efficiency Pure',
        'qb_quality': 0.20,
        'efficiency': 0.40,
        'line_play': 0.15,
        'situational': 0.15,
        'luck_regression': 0.10
    },
    'E_anti_luck': {
        'name': 'Anti-Luck',
        'qb_quality': 0.25,
        'efficiency': 0.20,
        'line_play': 0.15,
        'situational': 0.15,
        'luck_regression': 0.25
    }
}


class NFLModel:
    """Multi-profile NFL team scoring model"""

    def __init__(self):
        self.factors = None
        self.scores = None

    def load_factors(self) -> pd.DataFrame:
        """Load team factors from processed data"""
        factor_file = PROCESSED_DIR / "team_factors.csv"

        if not factor_file.exists():
            print(f"Error: {factor_file} not found. Run factor_calculator.py first.")
            return None

        self.factors = pd.read_csv(factor_file)
        print(f"Loaded factors for {len(self.factors)} teams")
        return self.factors

    def calculate_composite_score(self, row: pd.Series, weights: Dict) -> float:
        """Calculate weighted composite score for a team"""
        score = 0

        # Map factor column names to weight keys
        factor_map = {
            'qb_quality_score': 'qb_quality',
            'efficiency_score': 'efficiency',
            'line_play_score': 'line_play',
            'situational_score': 'situational',
            'luck_regression_score': 'luck_regression'
        }

        for col, weight_key in factor_map.items():
            if col in row.index:
                score += row[col] * weights.get(weight_key, 0)

        return score

    def score_all_teams(self) -> pd.DataFrame:
        """Score all teams using all weight profiles"""
        if self.factors is None:
            self.load_factors()

        if self.factors is None:
            return None

        print("Scoring teams with all weight profiles...")

        results = self.factors[['team']].copy()

        # Add primary QB if available
        if 'primary_qb' in self.factors.columns:
            results['primary_qb'] = self.factors['primary_qb']

        # Score with each profile
        for profile_key, profile in WEIGHT_PROFILES.items():
            profile_name = profile['name']
            weights = {k: v for k, v in profile.items() if k != 'name'}

            results[f'score_{profile_key}'] = self.factors.apply(
                lambda row: self.calculate_composite_score(row, weights),
                axis=1
            )

            print(f"  {profile_name}: Top team = {results.loc[results[f'score_{profile_key}'].idxmax(), 'team']}")

        # Calculate ensemble score (average across profiles)
        score_cols = [col for col in results.columns if col.startswith('score_')]
        results['score_ensemble'] = results[score_cols].mean(axis=1)

        # Rank teams
        results['rank_ensemble'] = results['score_ensemble'].rank(ascending=False).astype(int)

        # Sort by ensemble score
        results = results.sort_values('score_ensemble', ascending=False).reset_index(drop=True)

        self.scores = results
        return results

    def convert_to_win_probability(self, score: float, scale_factor: float = 0.02) -> float:
        """
        Convert composite score to win probability.
        Uses logistic function to bound between 0 and 1.

        Higher scale_factor = more extreme probabilities
        """
        # Center score around 50 (average)
        centered = score - 50

        # Logistic transformation
        prob = 1 / (1 + np.exp(-centered * scale_factor))

        return prob

    def calculate_championship_probabilities(self) -> pd.DataFrame:
        """
        Calculate Super Bowl championship probability for each team.
        Uses ensemble score as base, then normalizes.
        """
        if self.scores is None:
            self.score_all_teams()

        results = self.scores.copy()

        # Convert scores to raw probabilities
        results['raw_prob'] = results['score_ensemble'].apply(
            lambda x: self.convert_to_win_probability(x, scale_factor=0.05)
        )

        # Apply power transformation to increase separation between good and bad teams
        # This reflects that playoff format amplifies advantages
        results['power_prob'] = results['raw_prob'] ** 1.5

        # Normalize to sum to 1
        total = results['power_prob'].sum()
        results['championship_prob'] = results['power_prob'] / total

        # Express as percentage
        results['championship_pct'] = results['championship_prob'] * 100

        # Build column list dynamically based on what exists
        base_cols = ['team']
        if 'primary_qb' in results.columns:
            base_cols.append('primary_qb')
        base_cols.extend(['score_ensemble', 'rank_ensemble', 'championship_prob', 'championship_pct'])

        return results[base_cols +
                       [col for col in results.columns if col.startswith('score_') and col != 'score_ensemble']]

    def get_profile_comparison(self) -> pd.DataFrame:
        """Show how teams rank differently under each profile"""
        if self.scores is None:
            self.score_all_teams()

        comparison = self.scores[['team']].copy()

        for col in self.scores.columns:
            if col.startswith('score_'):
                rank_col = col.replace('score_', 'rank_')
                comparison[rank_col] = self.scores[col].rank(ascending=False).astype(int)

        return comparison

    def identify_model_divergence(self) -> pd.DataFrame:
        """
        Find teams where weight profiles disagree significantly.
        These may be interesting betting opportunities.
        """
        comparison = self.get_profile_comparison()

        rank_cols = [col for col in comparison.columns if col.startswith('rank_')]

        # Calculate rank variance
        comparison['rank_variance'] = comparison[rank_cols].var(axis=1)
        comparison['rank_std'] = comparison[rank_cols].std(axis=1)

        # High variance = profiles disagree = interesting
        comparison['divergence_score'] = comparison['rank_std']

        return comparison.sort_values('divergence_score', ascending=False)

    def run_model(self) -> Dict[str, pd.DataFrame]:
        """Run full model pipeline"""
        print("=" * 60)
        print("NFL SCORING MODEL")
        print("=" * 60)

        # Score teams
        scores = self.score_all_teams()

        if scores is None:
            return {}

        # Calculate probabilities
        probs = self.calculate_championship_probabilities()

        # Get divergence analysis
        divergence = self.identify_model_divergence()

        # Save outputs
        scores.to_csv(OUTPUTS_DIR / "team_scores.csv", index=False)
        probs.to_csv(OUTPUTS_DIR / "championship_probabilities.csv", index=False)
        divergence.to_csv(OUTPUTS_DIR / "model_divergence.csv", index=False)

        # Print results
        print("\n" + "=" * 60)
        print("TOP 10 TEAMS BY ENSEMBLE SCORE")
        print("=" * 60)
        print(probs[['team', 'primary_qb', 'score_ensemble', 'championship_pct']].head(10).to_string(index=False))

        print("\n" + "=" * 60)
        print("HIGHEST DIVERGENCE (profiles disagree)")
        print("=" * 60)
        print(divergence[['team', 'rank_std']].head(5).to_string(index=False))

        return {
            'scores': scores,
            'probabilities': probs,
            'divergence': divergence
        }


if __name__ == "__main__":
    model = NFLModel()
    results = model.run_model()
