# NFL Polymarket Model

A quantitative model for identifying mispriced Super Bowl futures on Polymarket. Uses FiveThirtyEight's Elo methodology, multi-factor analysis, and Monte Carlo simulation to find betting edges.

## Features

- **FiveThirtyEight Elo System**: Exact implementation with K=20, home advantage=48, playoff multiplier, and season regression
- **Unit-Specific Elo**: Separate ratings for Offense, Defense, and Special Teams
- **5-Factor Model**: QB Quality, Team Efficiency, Line Play, Situational, Luck Regression
- **5 Weight Profiles**: Baseline, QB Heavy, Line Heavy, Efficiency Pure, Anti-Luck
- **Monte Carlo Simulation**: 10,000 playoff bracket simulations
- **Kelly Criterion Sizing**: Fractional Kelly for position sizing
- **Web UI with Live Logging**: Real-time pipeline monitoring

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/nfl-polymarket-model.git
cd nfl-polymarket-model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run web UI
python app.py
# Open http://localhost:5000

# Or run from command line
python run_pipeline.py full    # Full pipeline with data pull
python run_pipeline.py quick   # Skip data pull
python run_pipeline.py edges   # Just check market edges
```

## Project Structure

```
nfl-polymarket-model/
├── app.py                 # Flask web UI with WebSocket logging
├── run_pipeline.py        # CLI pipeline runner
├── requirements.txt
├── src/
│   ├── data_pull.py       # NFL data fetching (nfl_data_py)
│   ├── elo_calculator.py  # FiveThirtyEight Elo implementation
│   ├── factor_calculator.py # 5-factor model
│   ├── model.py           # Scoring model with 5 weight profiles
│   ├── simulation.py      # Monte Carlo playoff simulation
│   ├── edge_detector.py   # Polymarket edge detection
│   ├── position_sizer.py  # Kelly criterion allocation
│   └── tracker.py         # Weekly tracking
├── templates/
│   └── index.html         # Dashboard UI
├── data/
│   ├── raw/               # Raw NFL data
│   └── processed/         # Processed factors and Elo ratings
└── outputs/
    ├── allocations.csv    # Recommended positions
    └── allocation_report.txt
```

## Model Methodology

### Elo System (FiveThirtyEight)
- Starting Elo: 1505
- K-factor: 20
- Home advantage: 48 points
- Playoff multiplier: 1.2x
- Season regression: 1/3 toward mean

### MOV Multiplier
```python
ln(|point_diff| + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
```

### Factor Weights (Baseline)
| Factor | Weight |
|--------|--------|
| QB Quality | 30% |
| Team Efficiency | 25% |
| Line Play | 20% |
| Situational | 15% |
| Luck Regression | 10% |

## Web UI

The dashboard provides:
- **Live Log Console**: Real-time pipeline output for debugging
- **Elo Rankings**: Current team Elo ratings
- **Monte Carlo Results**: Super Bowl win probabilities
- **Betting Edges**: Model vs market comparison
- **Allocations**: Kelly-sized position recommendations

## Data Sources

- **nfl_data_py**: Play-by-play, schedules, rosters
- **FiveThirtyEight**: Historical Elo baseline
- **Polymarket**: Market prices (API)

## License

MIT
