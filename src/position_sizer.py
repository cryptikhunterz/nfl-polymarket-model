"""
NFL Polymarket Position Sizer
Uses Kelly Criterion to allocate bankroll across betting opportunities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


class PositionSizer:
    """
    Kelly Criterion based position sizing for Polymarket bets.

    Kelly formula: f* = (bp - q) / b
    where:
        f* = fraction of bankroll to bet
        b = odds (payout ratio - 1)
        p = probability of winning (model probability)
        q = probability of losing (1 - p)

    For Polymarket:
        b = (1 / market_price) - 1
    """

    def __init__(self, bankroll: float = None, kelly_fraction: float = None):
        """
        Initialize position sizer.

        bankroll: Total USDC to allocate (default from .env or 100)
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly, more conservative)
        """
        self.bankroll = bankroll or float(os.getenv('BANKROLL', 100))
        self.kelly_fraction = kelly_fraction or float(os.getenv('KELLY_FRACTION', 0.25))

        print(f"Position Sizer initialized:")
        print(f"  Bankroll: ${self.bankroll:.2f} USDC")
        print(f"  Kelly Fraction: {self.kelly_fraction:.0%} (fractional Kelly)")

    def calculate_kelly(self, model_prob: float, market_price: float) -> float:
        """
        Calculate Kelly criterion bet size.

        Returns fraction of bankroll to bet (0 to 1).
        """
        if market_price <= 0 or market_price >= 1:
            return 0

        # Odds ratio (how much you win per dollar bet)
        b = (1 / market_price) - 1

        # Win/lose probability
        p = model_prob
        q = 1 - p

        # Kelly formula
        kelly = (b * p - q) / b

        # If negative or zero, don't bet
        if kelly <= 0:
            return 0

        # Apply fractional Kelly
        kelly_fractional = kelly * self.kelly_fraction

        # Cap at reasonable maximum (e.g., 25% of bankroll per bet)
        max_bet_fraction = 0.25
        kelly_fractional = min(kelly_fractional, max_bet_fraction)

        return kelly_fractional

    def size_positions(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position sizes for all opportunities.

        edges_df: DataFrame with 'team', 'model_prob', 'market_price', 'edge' columns
        """
        if edges_df is None or len(edges_df) == 0:
            print("No edges data to size")
            return pd.DataFrame()

        positions = edges_df.copy()

        # Calculate Kelly for each position
        positions['kelly_fraction'] = positions.apply(
            lambda row: self.calculate_kelly(row['model_prob'], row['market_price']),
            axis=1
        )

        # Calculate dollar amounts
        positions['kelly_amount'] = positions['kelly_fraction'] * self.bankroll

        # Filter to only positive positions
        positions = positions[positions['kelly_fraction'] > 0].copy()

        if len(positions) == 0:
            print("No positive Kelly positions found")
            return pd.DataFrame()

        # Normalize if total exceeds bankroll (diversification)
        total_kelly = positions['kelly_fraction'].sum()
        if total_kelly > 1:
            print(f"  Total Kelly ({total_kelly:.1%}) exceeds 100%, normalizing...")
            positions['normalized_fraction'] = positions['kelly_fraction'] / total_kelly
            positions['final_amount'] = positions['normalized_fraction'] * self.bankroll
        else:
            positions['normalized_fraction'] = positions['kelly_fraction']
            positions['final_amount'] = positions['kelly_amount']

        # Round to reasonable amounts (minimum $1, round to $0.50)
        positions['final_amount'] = positions['final_amount'].apply(
            lambda x: max(1, round(x * 2) / 2)  # Round to nearest $0.50, minimum $1
        )

        # Adjust if rounding pushed us over bankroll
        while positions['final_amount'].sum() > self.bankroll:
            # Reduce smallest position
            min_idx = positions['final_amount'].idxmin()
            if positions.loc[min_idx, 'final_amount'] <= 1:
                positions = positions.drop(min_idx)
            else:
                positions.loc[min_idx, 'final_amount'] -= 0.5

        # Calculate expected profit
        # E[profit] = amount * (1/price - 1) * prob - amount * (1 - prob)
        # Simplified: E[profit] = amount * EV
        positions['expected_profit'] = positions['final_amount'] * positions['ev']

        # Sort by expected profit
        positions = positions.sort_values('expected_profit', ascending=False).reset_index(drop=True)

        return positions

    def generate_allocation_report(self, positions: pd.DataFrame) -> str:
        """Generate human-readable allocation report"""
        if positions is None or len(positions) == 0:
            return "No allocations to report."

        report = []
        report.append("=" * 60)
        report.append("POLYMARKET ALLOCATION REPORT")
        report.append("=" * 60)
        report.append(f"Bankroll: ${self.bankroll:.2f} USDC")
        report.append(f"Kelly Fraction: {self.kelly_fraction:.0%}")
        report.append(f"Total Positions: {len(positions)}")
        report.append(f"Total Allocated: ${positions['final_amount'].sum():.2f}")
        report.append(f"Cash Reserve: ${self.bankroll - positions['final_amount'].sum():.2f}")
        report.append("")
        report.append("-" * 60)
        report.append("RECOMMENDED ALLOCATIONS:")
        report.append("-" * 60)

        for _, row in positions.iterrows():
            report.append(f"\n{row['team']}")
            report.append(f"  Allocation: ${row['final_amount']:.2f} USDC")
            report.append(f"  Model Prob: {row['model_prob']:.1%}")
            report.append(f"  Market Price: {row['market_price']:.1%}")
            report.append(f"  Edge: {row['edge_pct']:.1f}%")
            report.append(f"  Expected Profit: ${row['expected_profit']:.2f}")

        report.append("")
        report.append("-" * 60)
        report.append("SUMMARY:")
        report.append("-" * 60)
        report.append(f"Total Expected Profit: ${positions['expected_profit'].sum():.2f}")
        report.append(f"Expected ROI: {positions['expected_profit'].sum() / self.bankroll * 100:.1f}%")

        return "\n".join(report)


def size_positions():
    """Main function to size positions"""
    print("=" * 60)
    print("POSITION SIZING")
    print("=" * 60)

    # Load edges
    edges_file = OUTPUTS_DIR / "edges.csv"
    if not edges_file.exists():
        print("Error: edges.csv not found. Run edge_detector.py first.")
        return None

    edges = pd.read_csv(edges_file)

    # Filter to positive edge only
    edges = edges[edges['edge'] > 0]

    if len(edges) == 0:
        print("No positive edge opportunities found")
        return None

    # Size positions
    sizer = PositionSizer()
    positions = sizer.size_positions(edges)

    if len(positions) > 0:
        # Save allocations
        positions.to_csv(OUTPUTS_DIR / "allocations.csv", index=False)

        # Generate report
        report = sizer.generate_allocation_report(positions)
        print(report)

        # Save report
        with open(OUTPUTS_DIR / "allocation_report.txt", 'w') as f:
            f.write(report)

    return positions


if __name__ == "__main__":
    size_positions()
