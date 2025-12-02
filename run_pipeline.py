#!/usr/bin/env python3
"""
NFL Polymarket Model - Main Pipeline
Run the complete analysis from data pull to allocation recommendations
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import audit logger
from audit_logger import create_audit_logger


def run_full_pipeline(skip_data_pull: bool = False):
    """Run the complete pipeline"""
    # Create audit logger for new run
    logger = create_audit_logger()

    print("\n" + "=" * 60)
    print("NFL POLYMARKET MODEL - FULL PIPELINE")
    print("=" * 60 + "\n")
    print(f"Audit Logger initialized")

    # Step 1: Pull data
    if not skip_data_pull:
        print("\n[STEP 1/9] Pulling NFL Data...")
        print("-" * 40)
        from data_pull import pull_all_data
        pull_all_data()
    else:
        print("\n[STEP 1/9] Skipping data pull (using cached data)")

    # Step 2: Refresh starter/injury data
    print("\n[STEP 2/9] Refreshing Starter & Injury Data...")
    print("-" * 40)
    try:
        from starter_detector import refresh_all_data
        refresh_all_data()
    except Exception as e:
        print(f"Warning: Could not refresh starter data: {e}")
        print("Continuing with existing data...")

    # Step 3: Calculate Elo ratings
    print("\n[STEP 3/9] Calculating Elo Ratings...")
    print("-" * 40)
    from elo_calculator import calculate_all_elos
    calculate_all_elos()

    # Step 4: Calculate factors
    print("\n[STEP 4/9] Calculating Team Factors...")
    print("-" * 40)
    from factor_calculator import FactorCalculator
    calculator = FactorCalculator(current_season=2024)
    calculator.calculate_all_factors()

    # Step 5: Run model
    print("\n[STEP 5/9] Running Scoring Model...")
    print("-" * 40)
    from model import NFLModel
    model = NFLModel()
    model.run_model()

    # Step 6: Run simulation
    print("\n[STEP 6/9] Running Monte Carlo Simulation...")
    print("-" * 40)
    from simulation import run_monte_carlo
    run_monte_carlo()

    # Step 7: Detect edges
    print("\n[STEP 7/9] Detecting Market Edges...")
    print("-" * 40)
    from edge_detector import detect_edges
    detect_edges()

    # Step 8: Size positions
    print("\n[STEP 8/9] Sizing Positions...")
    print("-" * 40)
    from position_sizer import size_positions
    positions = size_positions()

    # Step 9: Generate audit report
    print("\n[STEP 9/9] Generating Audit Report...")
    print("-" * 40)
    try:
        logger.save_report()
        logger.print_summary()
    except Exception as e:
        print(f"Warning: Could not generate audit report: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nOutput files saved to: outputs/")
    print("  - team_scores.csv")
    print("  - championship_probabilities.csv")
    print("  - simulation_results.csv")
    print("  - edges.csv")
    print("  - allocations.csv")
    print("  - allocation_report.txt")
    print("  - audit_log.json")
    print("  - audit_report.md")

    return positions


def run_quick_update():
    """Run quick update (skip data pull, use cached data)"""
    return run_full_pipeline(skip_data_pull=True)


def run_edge_check():
    """Just check edges against current market prices"""
    print("\n" + "=" * 60)
    print("EDGE CHECK - Market Price Update")
    print("=" * 60 + "\n")

    from edge_detector import detect_edges
    from position_sizer import size_positions

    detect_edges()
    size_positions()


def run_weekly_tracking():
    """Run weekly tracking update"""
    print("\n" + "=" * 60)
    print("WEEKLY TRACKING UPDATE")
    print("=" * 60 + "\n")

    from tracker import track_week
    track_week()


def main():
    parser = argparse.ArgumentParser(description='NFL Polymarket Model Pipeline')
    parser.add_argument('command', nargs='?', default='full',
                        choices=['full', 'quick', 'edges', 'track'],
                        help='Command to run (default: full)')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data pull (use cached data)')

    args = parser.parse_args()

    if args.command == 'full':
        run_full_pipeline(skip_data_pull=args.skip_data)
    elif args.command == 'quick':
        run_quick_update()
    elif args.command == 'edges':
        run_edge_check()
    elif args.command == 'track':
        run_weekly_tracking()


if __name__ == "__main__":
    main()
