#!/usr/bin/env python3
"""
NFL Polymarket Model - Web UI with Real-time Logging
"""

import sys
import os
import json
import threading
import queue
from datetime import datetime
from pathlib import Path
from io import StringIO

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nfl-polymarket-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
pipeline_running = False
pipeline_thread = None
log_queue = queue.Queue()


class WebSocketLogHandler:
    """Custom log handler that sends logs to WebSocket clients"""

    def __init__(self, socketio_instance, original_stdout):
        self.socketio = socketio_instance
        self.original_stdout = original_stdout
        self.buffer = ""

    def write(self, message):
        # Write to original stdout
        self.original_stdout.write(message)
        self.original_stdout.flush()

        # Send to WebSocket
        if message.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = {
                'timestamp': timestamp,
                'message': message.rstrip(),
                'type': self._get_log_type(message)
            }
            self.socketio.emit('log', log_entry)

    def _get_log_type(self, message):
        msg_lower = message.lower()
        if 'error' in msg_lower or 'failed' in msg_lower:
            return 'error'
        elif 'warning' in msg_lower:
            return 'warning'
        elif '===' in message or '---' in message:
            return 'header'
        elif 'saved' in msg_lower or 'complete' in msg_lower:
            return 'success'
        return 'info'

    def flush(self):
        self.original_stdout.flush()


def run_pipeline_task(command):
    """Run pipeline in background thread"""
    global pipeline_running

    # Capture stdout
    original_stdout = sys.stdout
    sys.stdout = WebSocketLogHandler(socketio, original_stdout)

    try:
        pipeline_running = True
        socketio.emit('status', {'running': True, 'command': command})

        if command == 'full':
            from run_pipeline import run_full_pipeline
            run_full_pipeline(skip_data_pull=False)
        elif command == 'quick':
            from run_pipeline import run_quick_update
            run_quick_update()
        elif command == 'edges':
            from run_pipeline import run_edge_check
            run_edge_check()
        elif command == 'data':
            from data_pull import pull_all_data
            pull_all_data()
        elif command == 'elo':
            from elo_calculator import calculate_all_elos
            calculate_all_elos()
        elif command == 'factors':
            from factor_calculator import FactorCalculator
            calculator = FactorCalculator(current_season=2024)
            calculator.calculate_all_factors()
        elif command == 'model':
            from model import NFLModel
            model = NFLModel()
            model.run_model()
        elif command == 'simulate':
            from simulation import run_monte_carlo
            run_monte_carlo()
        elif command == 'starters':
            from starter_detector import refresh_all_data
            refresh_all_data()

        socketio.emit('pipeline_complete', {'success': True})

    except Exception as e:
        socketio.emit('pipeline_complete', {'success': False, 'error': str(e)})
        print(f"ERROR: {str(e)}")

    finally:
        sys.stdout = original_stdout
        pipeline_running = False
        socketio.emit('status', {'running': False})


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get current pipeline status"""
    return jsonify({
        'running': pipeline_running,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/run/<command>', methods=['POST'])
def run_command(command):
    """Run a pipeline command"""
    global pipeline_thread

    if pipeline_running:
        return jsonify({'error': 'Pipeline already running'}), 400

    valid_commands = ['full', 'quick', 'edges', 'data', 'elo', 'factors', 'model', 'simulate', 'starters']
    if command not in valid_commands:
        return jsonify({'error': f'Invalid command. Valid: {valid_commands}'}), 400

    pipeline_thread = threading.Thread(target=run_pipeline_task, args=(command,))
    pipeline_thread.daemon = True
    pipeline_thread.start()

    return jsonify({'status': 'started', 'command': command})


def clean_nan(obj):
    """Replace NaN values with None for JSON compatibility"""
    import pandas as pd
    import numpy as np
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    elif isinstance(obj, float) and (pd.isna(obj) or np.isnan(obj)):
        return None
    return obj


@app.route('/api/results')
def get_results():
    """Get latest results"""
    import pandas as pd
    outputs_dir = Path(__file__).parent / "outputs"
    results = {}

    # Read allocations
    alloc_file = outputs_dir / "allocations.csv"
    if alloc_file.exists():
        df = pd.read_csv(alloc_file)
        # Rename 'final_amount' to 'allocation' for frontend compatibility
        if 'final_amount' in df.columns:
            df = df.rename(columns={'final_amount': 'allocation'})
        results['allocations'] = clean_nan(df.to_dict('records'))

    # Read team scores
    scores_file = outputs_dir / "team_scores.csv"
    if scores_file.exists():
        df = pd.read_csv(scores_file)
        results['scores'] = clean_nan(df.head(15).to_dict('records'))

    # Read simulation results
    sim_file = outputs_dir / "simulation_results.csv"
    if sim_file.exists():
        df = pd.read_csv(sim_file)
        results['simulation'] = clean_nan(df.head(15).to_dict('records'))

    # Read edges
    edges_file = outputs_dir / "edges.csv"
    if edges_file.exists():
        df = pd.read_csv(edges_file)
        results['edges'] = clean_nan(df.to_dict('records'))

    # Read Elo ratings
    elo_file = Path(__file__).parent / "data" / "processed" / "team_elo.csv"
    if elo_file.exists():
        df = pd.read_csv(elo_file)
        results['elo'] = clean_nan(df.to_dict('records'))

    return jsonify(results)


@app.route('/api/report')
def get_report():
    """Get allocation report text"""
    report_file = Path(__file__).parent / "outputs" / "allocation_report.txt"
    if report_file.exists():
        return report_file.read_text()
    return "No report available. Run the pipeline first."


@app.route('/api/methodology')
def get_methodology():
    """Return explanation of model methodology"""
    from model import get_methodology
    return jsonify(get_methodology())


@app.route('/api/profiles')
def get_profiles():
    """Return all weight profiles"""
    from model import get_weight_profiles
    return jsonify(get_weight_profiles())


@app.route('/api/results/all-profiles')
def get_all_profile_results():
    """Return Super Bowl probabilities from all 5 profiles"""
    import pandas as pd
    from model import NFLModel

    try:
        model = NFLModel()
        model.load_factors()
        if model.factors is None:
            return jsonify({'error': 'No factors available. Run the pipeline first.'}), 404

        results = model.get_all_profile_probabilities()

        # Also load market prices for edge calculation
        edges_file = Path(__file__).parent / "outputs" / "edges.csv"
        market_prices = {}
        if edges_file.exists():
            edges_df = pd.read_csv(edges_file)
            for _, row in edges_df.iterrows():
                market_prices[row['team']] = row.get('market_prob', 0) * 100

        # Add market data and edge
        results_dict = clean_nan(results.to_dict('records'))
        for item in results_dict:
            team = item['team']
            market_price = market_prices.get(team, 0)
            item['market_price'] = round(market_price, 1)
            item['edge'] = round(item.get('prob_average', 0) - market_price, 1)

        return jsonify({
            'teams': results_dict,
            'profile_keys': ['baseline', 'qb_heavy', 'line_heavy', 'efficiency_pure', 'anti_luck']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/team/<team>/breakdown')
def get_team_breakdown(team):
    """Return detailed factor breakdown for a team"""
    import pandas as pd
    from model import NFLModel

    try:
        model = NFLModel()
        model.load_factors()
        if model.factors is None:
            return jsonify({'error': 'No factors available. Run the pipeline first.'}), 404

        breakdown = model.get_team_factor_breakdown(team.upper())

        if 'error' in breakdown:
            return jsonify(breakdown), 404

        # Add market price and probabilities
        edges_file = Path(__file__).parent / "outputs" / "edges.csv"
        if edges_file.exists():
            edges_df = pd.read_csv(edges_file)
            team_edge = edges_df[edges_df['team'] == team.upper()]
            if len(team_edge) > 0:
                breakdown['market_price'] = round(team_edge.iloc[0].get('market_prob', 0) * 100, 1)

        # Get probabilities from all profiles
        model.score_all_teams()
        all_probs = model.get_all_profile_probabilities()
        team_probs = all_probs[all_probs['team'] == team.upper()]
        if len(team_probs) > 0:
            prob_row = team_probs.iloc[0]
            breakdown['probabilities'] = {
                'baseline': prob_row.get('prob_baseline', 0),
                'qb_heavy': prob_row.get('prob_qb_heavy', 0),
                'line_heavy': prob_row.get('prob_line_heavy', 0),
                'efficiency_pure': prob_row.get('prob_efficiency_pure', 0),
                'anti_luck': prob_row.get('prob_anti_luck', 0),
                'average': prob_row.get('prob_average', 0),
                'spread': prob_row.get('prob_spread', 0)
            }
            breakdown['edge'] = round(prob_row.get('prob_average', 0) - breakdown.get('market_price', 0), 1)

        return jsonify(clean_nan(breakdown))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/market-prices')
def get_market_prices():
    """Return current market prices with full calculation chain for debugging"""
    import pandas as pd

    try:
        outputs_dir = Path(__file__).parent / "outputs"
        edges_file = outputs_dir / "edges.csv"
        alloc_file = outputs_dir / "allocations.csv"

        if not edges_file.exists():
            return jsonify({'error': 'No edges file. Run the pipeline first.'}), 404

        edges_df = pd.read_csv(edges_file)

        # Load allocations if available
        allocations = {}
        if alloc_file.exists():
            alloc_df = pd.read_csv(alloc_file)
            for _, row in alloc_df.iterrows():
                allocations[row['team']] = row.get('final_amount', row.get('allocation', 0))

        # Build market prices data
        market_data = []
        for _, row in edges_df.iterrows():
            team = row['team']
            model_prob = row.get('model_prob', 0)
            market_prob = row.get('market_price', row.get('implied_prob', 0))
            edge = row.get('edge', 0)
            kelly_frac = row.get('kelly_fraction', 0)

            market_data.append({
                'team': team,
                'model_prob': round(model_prob * 100, 1),
                'market_price': round(market_prob * 100, 1),
                'edge': round(edge * 100, 1),
                'kelly_pct': round(kelly_frac * 100, 2) if kelly_frac else 0,
                'allocation': round(allocations.get(team, 0), 2),
                'has_edge': edge > 0.02,
                'strong_edge': edge > 0.05
            })

        # Sort by edge descending
        market_data.sort(key=lambda x: x['edge'], reverse=True)

        return jsonify({
            'prices': market_data,
            'total_allocated': sum(allocations.values()),
            'positions_count': len([m for m in market_data if m['allocation'] > 0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kelly/<team>')
def get_kelly_calculation(team):
    """Show Kelly criterion calculation for a specific team"""
    import pandas as pd
    import math

    try:
        edges_file = Path(__file__).parent / "outputs" / "edges.csv"
        alloc_file = Path(__file__).parent / "outputs" / "allocations.csv"

        if not edges_file.exists():
            return jsonify({'error': 'No edges file. Run the pipeline first.'}), 404

        edges_df = pd.read_csv(edges_file)
        team_data = edges_df[edges_df['team'] == team.upper()]

        if len(team_data) == 0:
            return jsonify({'error': f'Team {team} not found'}), 404

        row = team_data.iloc[0]
        model_prob = row.get('model_prob', 0)
        market_prob = row.get('market_prob', 0)

        if market_prob <= 0:
            return jsonify({'error': 'Invalid market probability'}), 400

        # Kelly calculation
        implied_odds = 1 / market_prob
        edge = model_prob - market_prob

        # Kelly formula: (p * odds - 1) / (odds - 1)
        kelly_full = (model_prob * implied_odds - 1) / (implied_odds - 1) if implied_odds > 1 else 0
        kelly_half = kelly_full / 2
        kelly_quarter = kelly_full / 4

        # Get actual allocation if available
        allocation = 0
        if alloc_file.exists():
            alloc_df = pd.read_csv(alloc_file)
            team_alloc = alloc_df[alloc_df['team'] == team.upper()]
            if len(team_alloc) > 0:
                # Check for both column names (final_amount or allocation)
                allocation = team_alloc.iloc[0].get('final_amount', team_alloc.iloc[0].get('allocation', 0))

        return jsonify({
            'team': team.upper(),
            'model_probability': round(model_prob * 100, 1),
            'market_probability': round(market_prob * 100, 1),
            'edge': round(edge * 100, 1),
            'implied_odds': round(implied_odds, 2),
            'kelly_full': round(kelly_full * 100, 2),
            'kelly_half': round(kelly_half * 100, 2),
            'kelly_quarter': round(kelly_quarter * 100, 2),
            'recommended_allocation': round(allocation, 2),
            'formula': f'kelly = ({model_prob:.3f} * {implied_odds:.2f} - 1) / ({implied_odds:.2f} - 1) = {kelly_full:.4f}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/starters')
def get_starters():
    """Return current starter and injury data"""
    import pandas as pd

    try:
        outputs_dir = Path(__file__).parent / "outputs"
        starters_file = outputs_dir / "current_starters.csv"
        injuries_file = outputs_dir / "injuries.csv"

        result = {
            'starters': [],
            'injuries': [],
            'last_updated': None
        }

        if starters_file.exists():
            df = pd.read_csv(starters_file)
            result['starters'] = clean_nan(df.to_dict('records'))
            result['last_updated'] = starters_file.stat().st_mtime

        if injuries_file.exists():
            df = pd.read_csv(injuries_file)
            result['injuries'] = clean_nan(df.to_dict('records'))

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/audit')
def get_audit():
    """Return audit report summary and details"""
    try:
        outputs_dir = Path(__file__).parent / "outputs"
        audit_json = outputs_dir / "audit_log.json"
        audit_md = outputs_dir / "audit_report.md"

        result = {
            'available': False,
            'summary': None,
            'report_markdown': None,
            'last_updated': None
        }

        if audit_json.exists():
            result['available'] = True
            result['last_updated'] = audit_json.stat().st_mtime

            with open(audit_json, 'r') as f:
                audit_data = json.load(f)
                result['summary'] = audit_data.get('summary', {})
                result['api_calls'] = len(audit_data.get('api_calls', []))
                result['sanity_checks'] = audit_data.get('sanity_checks', [])
                result['warnings'] = audit_data.get('warnings', [])
                result['errors'] = audit_data.get('errors', [])

        if audit_md.exists():
            result['report_markdown'] = audit_md.read_text()

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/audit/report')
def get_audit_report():
    """Return the full audit report as Markdown text"""
    try:
        report_file = Path(__file__).parent / "outputs" / "audit_report.md"
        if report_file.exists():
            return report_file.read_text(), 200, {'Content-Type': 'text/markdown'}
        return "No audit report available. Run the pipeline first.", 404
    except Exception as e:
        return str(e), 500


# =====================
# Next Gen Stats Routes
# =====================

@app.route('/api/ngs/status')
def get_ngs_status():
    """Get Next Gen Stats data status"""
    try:
        stats_dir = Path(__file__).parent / "data" / "stats"
        season_file = stats_dir / "ngs_separation_season.csv"
        weekly_file = stats_dir / "ngs_separation_weekly.csv"
        team_ma_file = stats_dir / "team_separation_ma.csv"

        status = {
            'has_season_data': season_file.exists(),
            'has_weekly_data': weekly_file.exists(),
            'has_team_ma': team_ma_file.exists(),
            'last_updated': None,
            'season_rows': 0,
            'weekly_rows': 0
        }

        if season_file.exists():
            status['last_updated'] = datetime.fromtimestamp(season_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            import pandas as pd
            df = pd.read_csv(season_file)
            status['season_rows'] = len(df)

        if weekly_file.exists():
            import pandas as pd
            df = pd.read_csv(weekly_file)
            status['weekly_rows'] = len(df)

        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ngs/season')
def get_ngs_season():
    """Get Next Gen Stats season totals"""
    try:
        stats_dir = Path(__file__).parent / "data" / "stats"
        season_file = stats_dir / "ngs_separation_season.csv"

        if not season_file.exists():
            return jsonify({'error': 'No season data. Run scrape_stats.py first.'}), 404

        import pandas as pd
        df = pd.read_csv(season_file)
        return jsonify({
            'data': clean_nan(df.to_dict('records')),
            'count': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ngs/team-separation')
def get_ngs_team_separation():
    """Get team separation moving averages"""
    try:
        stats_dir = Path(__file__).parent / "data" / "stats"
        ma_file = stats_dir / "team_separation_ma.csv"

        if not ma_file.exists():
            return jsonify({'error': 'No team separation data. Run scrape_stats.py first.'}), 404

        import pandas as pd
        df = pd.read_csv(ma_file)
        # Remove empty rows
        df = df[df['team'].notna() & (df['team'] != '')]
        # Sort by separation (descending)
        df = df.sort_values('separation_4wk_ma', ascending=False)
        return jsonify({
            'data': clean_nan(df.to_dict('records')),
            'count': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================
# PFF Integration Routes
# =====================

@app.route('/api/pff/status')
def get_pff_status():
    """Get PFF integration status"""
    try:
        from pff_integration import get_pff_integration
        pff = get_pff_integration()
        status = pff.get_data_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pff/config', methods=['GET'])
def get_pff_config():
    """Get current PFF configuration"""
    try:
        config_path = Path(__file__).parent / "config" / "pff_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return jsonify(json.load(f))
        return jsonify({
            'data_source': 'mock',
            'api_key': None,
            'csv_path': None,
            'cache_hours': 24
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pff/config', methods=['POST'])
def save_pff_config():
    """Save PFF configuration"""
    try:
        data = request.json
        config_dir = Path(__file__).parent / "config"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "pff_config.json"

        config = {
            'data_source': data.get('data_source', 'mock'),
            'api_key': data.get('api_key'),
            'csv_path': data.get('csv_path'),
            'cache_hours': data.get('cache_hours', 24)
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return jsonify({'status': 'saved', 'config': config})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pff/metrics/<metric_type>')
def get_pff_metrics(metric_type):
    """Get PFF metrics (line_play or luck)"""
    try:
        from pff_integration import get_pff_integration
        pff = get_pff_integration()

        season = request.args.get('season', 2024, type=int)

        if metric_type == 'line_play':
            df = pff.get_line_play_metrics(season)
        elif metric_type == 'luck':
            df = pff.get_luck_regression_metrics(season)
        elif metric_type == 'all':
            df = pff.get_all_metrics(season)
        else:
            return jsonify({'error': f'Invalid metric type: {metric_type}'}), 400

        return jsonify({
            'metric_type': metric_type,
            'season': season,
            'data': clean_nan(df.to_dict('records'))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pff/upload/<metric_type>', methods=['POST'])
def upload_pff_csv(metric_type):
    """Upload CSV file for PFF metrics"""
    try:
        if metric_type not in ['line_play', 'luck']:
            return jsonify({'error': 'metric_type must be line_play or luck'}), 400

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save to temp location and process
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            file.save(tmp.name)

            from pff_integration import get_pff_integration
            pff = get_pff_integration()
            pff.upload_csv(tmp.name, metric_type)

            # Clean up
            os.unlink(tmp.name)

        return jsonify({'status': 'uploaded', 'metric_type': metric_type})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pff/test-connection', methods=['POST'])
def test_pff_connection():
    """Test PFF API connection with provided credentials"""
    try:
        data = request.json
        api_key = data.get('api_key')

        if not api_key:
            return jsonify({'error': 'No API key provided'}), 400

        # For now, PFF API is not implemented, so we return a mock response
        # In a real implementation, this would test the actual API connection
        return jsonify({
            'status': 'not_implemented',
            'message': 'PFF API integration requires subscription. Use CSV upload or mock data instead.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pff/upload-screenshots', methods=['POST'])
def upload_pff_screenshots():
    """
    Receive PFF screenshots and return extracted data.
    For now, this returns a template for manual data entry since
    OCR extraction would require additional dependencies.
    """
    try:
        if 'screenshots' not in request.files:
            return jsonify({'error': 'No screenshots provided'}), 400

        files = request.files.getlist('screenshots')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400

        # Save screenshots to temp directory for reference
        upload_dir = Path(__file__).parent / "data" / "uploads" / "pff_screenshots"
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for f in files:
            if f.filename:
                # Save with timestamp to avoid overwrites
                import time
                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}_{f.filename}"
                filepath = upload_dir / filename
                f.save(str(filepath))
                saved_files.append(filename)

        # Return template data structure for manual entry
        # The user will see this in the UI and can edit the values
        template_data = {
            'columns': ['team', 'record', 'pf', 'pa', 'overall', 'offense', 'pass', 'pblk', 'recv', 'run', 'rblk', 'defense', 'rdef', 'tack', 'prsh', 'cov', 'special'],
            'rows': [],
            'message': f'Screenshots saved ({len(saved_files)} files). Please enter the data from your screenshots below or edit the existing values.',
            'saved_files': saved_files
        }

        # Load existing data if available
        pff_grades_path = Path(__file__).parent / "data" / "stats" / "pff_team_grades_template.csv"
        if pff_grades_path.exists():
            import pandas as pd
            df = pd.read_csv(pff_grades_path)
            template_data['rows'] = df.to_dict('records')
            template_data['message'] = f'Screenshots saved ({len(saved_files)} files). Current PFF data loaded. Edit values as needed based on your screenshots.'

        # Return in format frontend expects
        return jsonify({
            'success': True,
            'files_processed': len(saved_files),
            'extracted_data': template_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pff/save-extracted', methods=['POST'])
def save_pff_extracted_data():
    """Save extracted/edited PFF data to CSV"""
    try:
        data = request.json
        if not data or 'data' not in data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        extracted = data['data']
        rows = extracted.get('rows', [])
        if not rows or len(rows) == 0:
            return jsonify({'success': False, 'error': 'Empty data'}), 400

        # Convert to DataFrame and save
        import pandas as pd
        df = pd.DataFrame(rows)

        # Ensure stats directory exists
        stats_dir = Path(__file__).parent / "data" / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # Save to the template file
        output_path = stats_dir / "pff_team_grades_template.csv"
        df.to_csv(output_path, index=False)

        # Also save a backup with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = stats_dir / f"pff_team_grades_{timestamp}.csv"
        df.to_csv(backup_path, index=False)

        return jsonify({
            'success': True,
            'path': str(output_path),
            'backup': str(backup_path),
            'rows_saved': len(rows)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status', {'running': pipeline_running})
    print(f"Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected")


if __name__ == '__main__':
    print("=" * 60)
    print("NFL POLYMARKET MODEL - WEB UI")
    print("=" * 60)
    print("\nStarting server at http://localhost:5050")
    print("Open this URL in your browser to access the dashboard\n")
    socketio.run(app, host='0.0.0.0', port=5050, debug=True, allow_unsafe_werkzeug=True)
