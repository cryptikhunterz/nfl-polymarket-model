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
    'baseline': {
        'name': 'Baseline',
        'description': 'Balanced starting point',
        'qb_quality': 0.30,
        'efficiency': 0.25,
        'line_play': 0.20,
        'situational': 0.15,
        'luck_regression': 0.10
    },
    'qb_heavy': {
        'name': 'QB Heavy',
        'description': 'Playoffs are QB battles',
        'qb_quality': 0.45,
        'efficiency': 0.20,
        'line_play': 0.15,
        'situational': 0.12,
        'luck_regression': 0.08
    },
    'line_heavy': {
        'name': 'Line Heavy',
        'description': 'OL is undervalued by market',
        'qb_quality': 0.25,
        'efficiency': 0.20,
        'line_play': 0.30,
        'situational': 0.15,
        'luck_regression': 0.10
    },
    'efficiency_pure': {
        'name': 'Efficiency Pure',
        'description': 'Trust Elo and EPA most',
        'qb_quality': 0.20,
        'efficiency': 0.40,
        'line_play': 0.15,
        'situational': 0.15,
        'luck_regression': 0.10
    },
    'anti_luck': {
        'name': 'Anti-Luck',
        'description': 'Market ignores regression',
        'qb_quality': 0.25,
        'efficiency': 0.20,
        'line_play': 0.15,
        'situational': 0.15,
        'luck_regression': 0.25
    }
}

# Methodology explanation
METHODOLOGY = {
    'team_elo': {
        'source': 'FiveThirtyEight',
        'k_factor': 20,
        'start_elo': 1505,
        'home_advantage': 48,
        'season_regression': '1/3 toward 1505',
        'mov_formula': 'ln(PD+1) * (2.2 / (elo_diff * 0.001 + 2.2))',
        'playoff_multiplier': 1.2
    },
    'unit_elo': {
        'source': 'Custom',
        'offensive': 'Points scored vs opponent defensive Elo',
        'defensive': 'Points allowed vs opponent offensive Elo',
        'special_teams': 'FG%, return yards, field position'
    },
    'data_lookback': '1.5 seasons (70% current, 30% last)',
    'rolling_window': '5-game average weighted against season average'
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
            weights = {k: v for k, v in profile.items() if k not in ['name', 'description']}

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

    def get_all_profile_probabilities(self) -> pd.DataFrame:
        """
        Calculate Super Bowl probabilities for ALL weight profiles.
        Returns a table comparing all profiles side by side.
        """
        if self.scores is None:
            self.score_all_teams()

        results = self.scores[['team']].copy()

        # Add primary QB if available
        if 'primary_qb' in self.scores.columns:
            results['primary_qb'] = self.scores['primary_qb']

        # Calculate probabilities for each profile
        for profile_key in WEIGHT_PROFILES.keys():
            score_col = f'score_{profile_key}'
            if score_col in self.scores.columns:
                # Convert scores to probabilities
                raw_probs = self.scores[score_col].apply(
                    lambda x: self.convert_to_win_probability(x, scale_factor=0.05)
                )
                # Power transformation and normalize
                power_probs = raw_probs ** 1.5
                total = power_probs.sum()
                results[f'prob_{profile_key}'] = (power_probs / total * 100).round(1)

        # Calculate average probability across profiles
        prob_cols = [col for col in results.columns if col.startswith('prob_')]
        results['prob_average'] = results[prob_cols].mean(axis=1).round(1)

        # Calculate spread (max - min) as uncertainty measure
        results['prob_spread'] = (results[prob_cols].max(axis=1) - results[prob_cols].min(axis=1)).round(1)

        # Sort by average probability
        results = results.sort_values('prob_average', ascending=False).reset_index(drop=True)

        return results

    def get_team_factor_breakdown(self, team: str) -> Dict:
        """
        Get detailed factor breakdown for a specific team.
        Shows how each factor contributes to the team's score.
        """
        if self.factors is None:
            self.load_factors()

        if self.factors is None:
            return {'error': 'No factors available'}

        team_row = self.factors[self.factors['team'] == team]
        if len(team_row) == 0:
            return {'error': f'Team {team} not found'}

        team_row = team_row.iloc[0]

        # Factor details
        factor_details = []
        factor_cols = {
            'qb_quality_score': {'name': 'QB Quality', 'weight_key': 'qb_quality'},
            'efficiency_score': {'name': 'Team Efficiency', 'weight_key': 'efficiency'},
            'line_play_score': {'name': 'Line Play', 'weight_key': 'line_play'},
            'situational_score': {'name': 'Situational', 'weight_key': 'situational'},
            'luck_regression_score': {'name': 'Luck Regression', 'weight_key': 'luck_regression'}
        }

        for col, info in factor_cols.items():
            if col in team_row.index:
                score = team_row[col]
                factor_details.append({
                    'factor': info['name'],
                    'score': round(score, 1) if pd.notna(score) else 0,
                    'normalized': round(score, 0) if pd.notna(score) else 0,
                    'weight_key': info['weight_key']
                })

        # Calculate contributions by profile
        contributions_by_profile = {}
        for profile_key, profile in WEIGHT_PROFILES.items():
            profile_contributions = []
            weights = {k: v for k, v in profile.items() if k not in ['name', 'description']}
            total_contribution = 0

            for factor in factor_details:
                weight = weights.get(factor['weight_key'], 0)
                contribution = factor['score'] * weight
                total_contribution += contribution
                profile_contributions.append({
                    'factor': factor['factor'],
                    'score': factor['score'],
                    'weight': f"{int(weight * 100)}%",
                    'contribution': round(contribution, 1)
                })

            contributions_by_profile[profile_key] = {
                'name': profile['name'],
                'factors': profile_contributions,
                'total_score': round(total_contribution, 1)
            }

        return {
            'team': team,
            'primary_qb': team_row.get('primary_qb', 'N/A'),
            'factor_scores': factor_details,
            'contributions_by_profile': contributions_by_profile
        }


def get_methodology():
    """Return the methodology explanation"""
    return METHODOLOGY


def get_weight_profiles():
    """Return all weight profiles"""
    return WEIGHT_PROFILES


if __name__ == "__main__":
    model = NFLModel()
    results = model.run_model()
