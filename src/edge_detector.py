"""
NFL Polymarket Edge Detector
Compares model probabilities to Polymarket prices to find mispriced futures
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
POLYMARKET_DIR = Path(__file__).parent.parent / "data" / "polymarket"
POLYMARKET_DIR.mkdir(parents=True, exist_ok=True)


class PolymarketFetcher:
    """Fetch Super Bowl futures prices from Polymarket"""

    # Polymarket API endpoints
    BASE_URL = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"

    # Known Super Bowl market slugs (update for current season)
    SUPER_BOWL_SLUGS = [
        "super-bowl-lix-winner",
        "nfl-super-bowl-winner",
        "who-will-win-super-bowl"
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'NFL-Model/1.0'
        })

    def search_markets(self, query: str = "super bowl") -> List[Dict]:
        """Search for markets matching query"""
        try:
            url = f"{self.GAMMA_API}/markets"
            params = {'search': query, 'limit': 50}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching markets: {e}")
            return []

    def get_market_by_slug(self, slug: str) -> Optional[Dict]:
        """Get market details by slug"""
        try:
            url = f"{self.GAMMA_API}/markets/{slug}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching market {slug}: {e}")
            return None

    def get_super_bowl_prices(self) -> pd.DataFrame:
        """
        Fetch current Super Bowl futures prices.
        Returns DataFrame with team, price (implied probability), and market info.
        """
        print("Fetching Polymarket Super Bowl prices...")

        # Try known slugs first
        market_data = None
        for slug in self.SUPER_BOWL_SLUGS:
            market_data = self.get_market_by_slug(slug)
            if market_data:
                print(f"  Found market: {slug}")
                break

        # If no known slug works, search
        if not market_data:
            markets = self.search_markets("super bowl")
            if markets:
                # Find the NFL Super Bowl winner market
                for m in markets:
                    if 'super bowl' in m.get('question', '').lower():
                        market_data = m
                        print(f"  Found market via search: {m.get('slug')}")
                        break

        if not market_data:
            print("  Could not find Super Bowl market")
            return self._get_fallback_prices()

        # Parse market data into prices
        prices = self._parse_market_prices(market_data)
        return prices

    def _parse_market_prices(self, market_data: Dict) -> pd.DataFrame:
        """Parse market data into team prices"""
        prices = []

        # Polymarket markets can have different structures
        outcomes = market_data.get('outcomes', [])
        tokens = market_data.get('tokens', [])

        if outcomes:
            for outcome in outcomes:
                team_name = outcome.get('name', outcome.get('title', ''))
                price = outcome.get('price', outcome.get('probability', 0))

                # Convert price to float
                if isinstance(price, str):
                    price = float(price)

                # Normalize team name to abbreviation
                team_abbr = self._normalize_team_name(team_name)

                if team_abbr:
                    prices.append({
                        'team': team_abbr,
                        'team_name': team_name,
                        'market_price': price,
                        'implied_prob': price,  # Polymarket price IS the probability
                        'implied_prob_pct': price * 100
                    })

        elif tokens:
            for token in tokens:
                team_name = token.get('outcome', '')
                price = token.get('price', 0)

                team_abbr = self._normalize_team_name(team_name)

                if team_abbr:
                    prices.append({
                        'team': team_abbr,
                        'team_name': team_name,
                        'market_price': float(price),
                        'implied_prob': float(price),
                        'implied_prob_pct': float(price) * 100
                    })

        if not prices:
            return self._get_fallback_prices()

        df = pd.DataFrame(prices)
        df = df.sort_values('implied_prob', ascending=False).reset_index(drop=True)

        # Save to file
        df.to_csv(POLYMARKET_DIR / "super_bowl_prices.csv", index=False)
        print(f"  Saved prices for {len(df)} teams")

        return df

    def _normalize_team_name(self, name: str) -> Optional[str]:
        """Convert team name to standard abbreviation"""
        name_lower = name.lower()

        team_map = {
            'kansas city chiefs': 'KC', 'chiefs': 'KC', 'kansas city': 'KC',
            'san francisco 49ers': 'SF', '49ers': 'SF', 'san francisco': 'SF', 'niners': 'SF',
            'buffalo bills': 'BUF', 'bills': 'BUF', 'buffalo': 'BUF',
            'philadelphia eagles': 'PHI', 'eagles': 'PHI', 'philadelphia': 'PHI',
            'detroit lions': 'DET', 'lions': 'DET', 'detroit': 'DET',
            'baltimore ravens': 'BAL', 'ravens': 'BAL', 'baltimore': 'BAL',
            'dallas cowboys': 'DAL', 'cowboys': 'DAL', 'dallas': 'DAL',
            'miami dolphins': 'MIA', 'dolphins': 'MIA', 'miami': 'MIA',
            'green bay packers': 'GB', 'packers': 'GB', 'green bay': 'GB',
            'cincinnati bengals': 'CIN', 'bengals': 'CIN', 'cincinnati': 'CIN',
            'los angeles rams': 'LAR', 'rams': 'LAR',
            'los angeles chargers': 'LAC', 'chargers': 'LAC',
            'new york jets': 'NYJ', 'jets': 'NYJ',
            'new york giants': 'NYG', 'giants': 'NYG',
            'seattle seahawks': 'SEA', 'seahawks': 'SEA', 'seattle': 'SEA',
            'minnesota vikings': 'MIN', 'vikings': 'MIN', 'minnesota': 'MIN',
            'jacksonville jaguars': 'JAX', 'jaguars': 'JAX', 'jacksonville': 'JAX',
            'cleveland browns': 'CLE', 'browns': 'CLE', 'cleveland': 'CLE',
            'pittsburgh steelers': 'PIT', 'steelers': 'PIT', 'pittsburgh': 'PIT',
            'houston texans': 'HOU', 'texans': 'HOU', 'houston': 'HOU',
            'indianapolis colts': 'IND', 'colts': 'IND', 'indianapolis': 'IND',
            'las vegas raiders': 'LV', 'raiders': 'LV', 'las vegas': 'LV',
            'denver broncos': 'DEN', 'broncos': 'DEN', 'denver': 'DEN',
            'tennessee titans': 'TEN', 'titans': 'TEN', 'tennessee': 'TEN',
            'atlanta falcons': 'ATL', 'falcons': 'ATL', 'atlanta': 'ATL',
            'new orleans saints': 'NO', 'saints': 'NO', 'new orleans': 'NO',
            'tampa bay buccaneers': 'TB', 'buccaneers': 'TB', 'tampa bay': 'TB', 'bucs': 'TB',
            'carolina panthers': 'CAR', 'panthers': 'CAR', 'carolina': 'CAR',
            'chicago bears': 'CHI', 'bears': 'CHI', 'chicago': 'CHI',
            'arizona cardinals': 'ARI', 'cardinals': 'ARI', 'arizona': 'ARI',
            'new england patriots': 'NE', 'patriots': 'NE', 'new england': 'NE',
            'washington commanders': 'WAS', 'commanders': 'WAS', 'washington': 'WAS',
        }

        for key, abbr in team_map.items():
            if key in name_lower:
                return abbr

        return None

    def _get_fallback_prices(self) -> pd.DataFrame:
        """
        Return fallback estimated prices if API fails.
        Based on typical Super Bowl futures odds.
        """
        print("  Using fallback estimated prices")

        # Typical mid-season Super Bowl odds (update as needed)
        fallback = [
            {'team': 'KC', 'implied_prob_pct': 15.0},
            {'team': 'SF', 'implied_prob_pct': 12.0},
            {'team': 'DET', 'implied_prob_pct': 10.0},
            {'team': 'BAL', 'implied_prob_pct': 9.0},
            {'team': 'PHI', 'implied_prob_pct': 8.0},
            {'team': 'BUF', 'implied_prob_pct': 7.0},
            {'team': 'GB', 'implied_prob_pct': 5.0},
            {'team': 'MIN', 'implied_prob_pct': 4.0},
            {'team': 'HOU', 'implied_prob_pct': 4.0},
            {'team': 'CIN', 'implied_prob_pct': 3.5},
            {'team': 'MIA', 'implied_prob_pct': 3.0},
            {'team': 'DAL', 'implied_prob_pct': 3.0},
            {'team': 'LAC', 'implied_prob_pct': 2.5},
            {'team': 'PIT', 'implied_prob_pct': 2.0},
            {'team': 'SEA', 'implied_prob_pct': 2.0},
            {'team': 'DEN', 'implied_prob_pct': 1.5},
            {'team': 'TB', 'implied_prob_pct': 1.5},
            {'team': 'LAR', 'implied_prob_pct': 1.5},
            {'team': 'ATL', 'implied_prob_pct': 1.0},
            {'team': 'NYJ', 'implied_prob_pct': 1.0},
            {'team': 'JAX', 'implied_prob_pct': 0.8},
            {'team': 'IND', 'implied_prob_pct': 0.6},
            {'team': 'CLE', 'implied_prob_pct': 0.5},
            {'team': 'NO', 'implied_prob_pct': 0.5},
            {'team': 'ARI', 'implied_prob_pct': 0.3},
            {'team': 'CHI', 'implied_prob_pct': 0.3},
            {'team': 'NYG', 'implied_prob_pct': 0.2},
            {'team': 'TEN', 'implied_prob_pct': 0.2},
            {'team': 'LV', 'implied_prob_pct': 0.2},
            {'team': 'WAS', 'implied_prob_pct': 0.2},
            {'team': 'NE', 'implied_prob_pct': 0.1},
            {'team': 'CAR', 'implied_prob_pct': 0.1},
        ]

        df = pd.DataFrame(fallback)
        df['implied_prob'] = df['implied_prob_pct'] / 100
        df['market_price'] = df['implied_prob']
        df['team_name'] = df['team']  # Placeholder

        df.to_csv(POLYMARKET_DIR / "super_bowl_prices.csv", index=False)
        return df


class EdgeDetector:
    """Detect edges between model probabilities and market prices"""

    def __init__(self, min_edge: float = 0.02):
        """
        Initialize edge detector.

        min_edge: Minimum edge (model - market) to consider a bet
        """
        self.min_edge = min_edge
        self.model_probs = None
        self.market_prices = None

    def load_data(self):
        """Load model probabilities and market prices"""
        # Load simulation results (most accurate model output)
        sim_file = OUTPUTS_DIR / "simulation_results.csv"
        if sim_file.exists():
            self.model_probs = pd.read_csv(sim_file)
            # Convert to decimal
            if 'sb_win_pct' in self.model_probs.columns:
                self.model_probs['model_prob'] = self.model_probs['sb_win_pct'] / 100
            print(f"Loaded model probabilities for {len(self.model_probs)} teams")
        else:
            # Fall back to model probabilities
            prob_file = OUTPUTS_DIR / "championship_probabilities.csv"
            if prob_file.exists():
                self.model_probs = pd.read_csv(prob_file)
                self.model_probs['model_prob'] = self.model_probs['championship_pct'] / 100
                print(f"Loaded model probabilities from championship file")

        # Load market prices
        price_file = POLYMARKET_DIR / "super_bowl_prices.csv"
        if price_file.exists():
            self.market_prices = pd.read_csv(price_file)
        else:
            # Fetch fresh prices
            fetcher = PolymarketFetcher()
            self.market_prices = fetcher.get_super_bowl_prices()

    def calculate_edges(self) -> pd.DataFrame:
        """Calculate edge for each team"""
        if self.model_probs is None or self.market_prices is None:
            self.load_data()

        if self.model_probs is None:
            print("Error: No model probabilities found")
            return pd.DataFrame()

        # Merge model and market data
        edges = self.model_probs[['team', 'model_prob']].merge(
            self.market_prices[['team', 'implied_prob', 'market_price']],
            on='team',
            how='inner'
        )

        # Calculate edge (positive = model likes team more than market)
        edges['edge'] = edges['model_prob'] - edges['implied_prob']
        edges['edge_pct'] = edges['edge'] * 100

        # Calculate expected value (EV)
        # EV = (win_prob * payout) - (loss_prob * stake)
        # For $1 bet at price p: payout = 1/p if win, lose $1 if lose
        # EV = model_prob * (1/market_price - 1) - (1 - model_prob) * 1
        # Simplified: EV = model_prob / market_price - 1
        edges['ev'] = edges['model_prob'] / edges['market_price'] - 1
        edges['ev_pct'] = edges['ev'] * 100

        # Flag bets with positive edge above threshold
        edges['has_edge'] = edges['edge'] >= self.min_edge

        # Sort by edge
        edges = edges.sort_values('edge', ascending=False).reset_index(drop=True)

        # Save
        edges.to_csv(OUTPUTS_DIR / "edges.csv", index=False)

        return edges

    def get_betting_opportunities(self) -> pd.DataFrame:
        """Get teams with positive edge above threshold"""
        edges = self.calculate_edges()

        opportunities = edges[edges['has_edge']].copy()

        print("\n" + "=" * 60)
        print("BETTING OPPORTUNITIES (Edge >= {:.1%})".format(self.min_edge))
        print("=" * 60)

        if len(opportunities) == 0:
            print("No opportunities found above threshold")
        else:
            print(opportunities[['team', 'model_prob', 'implied_prob', 'edge_pct', 'ev_pct']].to_string(index=False))

        return opportunities


def detect_edges():
    """Main function to detect edges"""
    # First fetch market prices
    fetcher = PolymarketFetcher()
    fetcher.get_super_bowl_prices()

    # Then calculate edges
    detector = EdgeDetector(min_edge=0.02)  # 2% minimum edge
    edges = detector.calculate_edges()

    opportunities = detector.get_betting_opportunities()
    return edges, opportunities


if __name__ == "__main__":
    detect_edges()
