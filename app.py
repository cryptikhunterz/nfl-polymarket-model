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

    valid_commands = ['full', 'quick', 'edges', 'data', 'elo', 'factors', 'model', 'simulate']
    if command not in valid_commands:
        return jsonify({'error': f'Invalid command. Valid: {valid_commands}'}), 400

    pipeline_thread = threading.Thread(target=run_pipeline_task, args=(command,))
    pipeline_thread.daemon = True
    pipeline_thread.start()

    return jsonify({'status': 'started', 'command': command})


@app.route('/api/results')
def get_results():
    """Get latest results"""
    outputs_dir = Path(__file__).parent / "outputs"
    results = {}

    # Read allocations
    alloc_file = outputs_dir / "allocations.csv"
    if alloc_file.exists():
        import pandas as pd
        df = pd.read_csv(alloc_file)
        results['allocations'] = df.to_dict('records')

    # Read team scores
    scores_file = outputs_dir / "team_scores.csv"
    if scores_file.exists():
        import pandas as pd
        df = pd.read_csv(scores_file)
        results['scores'] = df.head(15).to_dict('records')

    # Read simulation results
    sim_file = outputs_dir / "simulation_results.csv"
    if sim_file.exists():
        import pandas as pd
        df = pd.read_csv(sim_file)
        results['simulation'] = df.head(15).to_dict('records')

    # Read edges
    edges_file = outputs_dir / "edges.csv"
    if edges_file.exists():
        import pandas as pd
        df = pd.read_csv(edges_file)
        results['edges'] = df.to_dict('records')

    # Read Elo ratings
    elo_file = Path(__file__).parent / "data" / "processed" / "team_elo.csv"
    if elo_file.exists():
        import pandas as pd
        df = pd.read_csv(elo_file)
        results['elo'] = df.to_dict('records')

    return jsonify(results)


@app.route('/api/report')
def get_report():
    """Get allocation report text"""
    report_file = Path(__file__).parent / "outputs" / "allocation_report.txt"
    if report_file.exists():
        return report_file.read_text()
    return "No report available. Run the pipeline first."


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
