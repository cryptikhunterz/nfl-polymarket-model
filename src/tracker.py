"""
NFL Polymarket Weekly Tracker
Tracks positions, price changes, and model performance over time
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


class WeeklyTracker:
    """Track positions and performance week over week"""

    def __init__(self):
        self.tracking_file = OUTPUTS_DIR / "tracking.csv"
        self.history = self._load_history()

    def _load_history(self) -> pd.DataFrame:
        """Load existing tracking history"""
        if self.tracking_file.exists():
            return pd.read_csv(self.tracking_file)
        return pd.DataFrame()

    def _save_history(self):
        """Save tracking history"""
        if len(self.history) > 0:
            self.history.to_csv(self.tracking_file, index=False)

    def record_snapshot(self, allocations: pd.DataFrame, edges: pd.DataFrame,
                        week: int = None, season: int = 2024) -> pd.DataFrame:
        """
        Record current state as a weekly snapshot.

        allocations: Current position allocations
        edges: Current edge calculations
        week: NFL week number (auto-detected if None)
        """
        timestamp = datetime.now().isoformat()

        if week is None:
            # Estimate week based on date (NFL season starts ~Sep 5)
            from datetime import date
            today = date.today()
            season_start = date(2024, 9, 5)
            days_since_start = (today - season_start).days
            week = max(1, min(18, days_since_start // 7 + 1))

        snapshot_rows = []

        # Merge allocations with edges for full picture
        if len(allocations) > 0 and len(edges) > 0:
            data = allocations.merge(edges[['team', 'model_prob', 'market_price', 'edge']],
                                     on='team', how='left', suffixes=('', '_edge'))

            for _, row in data.iterrows():
                snapshot_rows.append({
                    'timestamp': timestamp,
                    'season': season,
                    'week': week,
                    'team': row['team'],
                    'allocation': row.get('final_amount', 0),
                    'model_prob': row.get('model_prob', 0),
                    'market_price': row.get('market_price', 0),
                    'edge': row.get('edge', 0),
                    'expected_profit': row.get('expected_profit', 0)
                })
        else:
            # Record edges only
            for _, row in edges.iterrows():
                snapshot_rows.append({
                    'timestamp': timestamp,
                    'season': season,
                    'week': week,
                    'team': row['team'],
                    'allocation': 0,
                    'model_prob': row.get('model_prob', 0),
                    'market_price': row.get('implied_prob', row.get('market_price', 0)),
                    'edge': row.get('edge', 0),
                    'expected_profit': 0
                })

        snapshot = pd.DataFrame(snapshot_rows)

        # Append to history
        if len(self.history) == 0:
            self.history = snapshot
        else:
            self.history = pd.concat([self.history, snapshot], ignore_index=True)

        self._save_history()
        print(f"Recorded snapshot for Week {week}")

        return snapshot

    def get_price_changes(self, team: str = None) -> pd.DataFrame:
        """Get price changes over time for a team or all teams"""
        if len(self.history) == 0:
            return pd.DataFrame()

        if team:
            data = self.history[self.history['team'] == team].copy()
        else:
            data = self.history.copy()

        # Calculate week-over-week changes
        data = data.sort_values(['team', 'week'])
        data['price_change'] = data.groupby('team')['market_price'].diff()
        data['edge_change'] = data.groupby('team')['edge'].diff()

        return data

    def get_team_history(self, team: str) -> pd.DataFrame:
        """Get full tracking history for a specific team"""
        if len(self.history) == 0:
            return pd.DataFrame()

        return self.history[self.history['team'] == team].sort_values('week')

    def calculate_realized_pnl(self, actual_results: Dict[str, bool]) -> pd.DataFrame:
        """
        Calculate realized P&L based on actual Super Bowl result.

        actual_results: Dict mapping team to whether they won (True/False)
        Example: {'KC': True, 'SF': False, ...}
        """
        if len(self.history) == 0:
            return pd.DataFrame()

        # Get most recent allocations
        latest_week = self.history['week'].max()
        final_positions = self.history[self.history['week'] == latest_week].copy()

        pnl_rows = []
        for _, row in final_positions.iterrows():
            team = row['team']
            allocation = row['allocation']

            if allocation > 0 and team in actual_results:
                if actual_results[team]:
                    # Won: profit = allocation * (1/price - 1)
                    payout = allocation / row['market_price']
                    profit = payout - allocation
                else:
                    # Lost: lose allocation
                    profit = -allocation

                pnl_rows.append({
                    'team': team,
                    'allocation': allocation,
                    'market_price': row['market_price'],
                    'model_prob': row['model_prob'],
                    'won': actual_results[team],
                    'profit': profit,
                    'roi': profit / allocation if allocation > 0 else 0
                })

        return pd.DataFrame(pnl_rows)

    def generate_weekly_report(self) -> str:
        """Generate weekly tracking report"""
        if len(self.history) == 0:
            return "No tracking history available."

        latest_week = self.history['week'].max()
        current = self.history[self.history['week'] == latest_week]

        report = []
        report.append("=" * 60)
        report.append(f"WEEKLY TRACKING REPORT - Week {latest_week}")
        report.append("=" * 60)

        # Current positions
        positions = current[current['allocation'] > 0]
        if len(positions) > 0:
            report.append("\nCurrent Positions:")
            report.append("-" * 40)
            for _, row in positions.iterrows():
                report.append(f"  {row['team']}: ${row['allocation']:.2f}")
                report.append(f"    Edge: {row['edge']*100:.1f}%")
                report.append(f"    Model: {row['model_prob']*100:.1f}% vs Market: {row['market_price']*100:.1f}%")

            total_allocated = positions['allocation'].sum()
            total_expected = positions['expected_profit'].sum()
            report.append(f"\nTotal Allocated: ${total_allocated:.2f}")
            report.append(f"Total Expected Profit: ${total_expected:.2f}")

        # Price changes
        if latest_week > 1:
            changes = self.get_price_changes()
            week_changes = changes[changes['week'] == latest_week]

            if len(week_changes) > 0:
                report.append("\nPrice Changes This Week:")
                report.append("-" * 40)

                significant = week_changes[abs(week_changes['price_change']) > 0.01]
                for _, row in significant.nlargest(5, 'price_change').iterrows():
                    direction = "+" if row['price_change'] > 0 else ""
                    report.append(f"  {row['team']}: {direction}{row['price_change']*100:.1f}%")

        return "\n".join(report)


def track_week():
    """Main function to record weekly tracking"""
    print("=" * 60)
    print("WEEKLY TRACKING")
    print("=" * 60)

    # Load current data
    alloc_file = OUTPUTS_DIR / "allocations.csv"
    edges_file = OUTPUTS_DIR / "edges.csv"

    allocations = pd.read_csv(alloc_file) if alloc_file.exists() else pd.DataFrame()
    edges = pd.read_csv(edges_file) if edges_file.exists() else pd.DataFrame()

    if len(edges) == 0:
        print("Error: No edges data found. Run edge_detector.py first.")
        return

    # Record snapshot
    tracker = WeeklyTracker()
    tracker.record_snapshot(allocations, edges)

    # Generate report
    report = tracker.generate_weekly_report()
    print(report)

    # Save report
    with open(OUTPUTS_DIR / "weekly_report.txt", 'w') as f:
        f.write(report)


if __name__ == "__main__":
    track_week()
