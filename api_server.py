"""
Mechano-Velocity REST API Server

Flask-based API that exposes the analysis pipeline to the frontend.
"""

import os
import json
import shutil
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from pipeline_runner import PipelineRunner
from mechano_velocity import Config, DatabaseManager

# ----------------------------------------------------------------
# App Setup
# ----------------------------------------------------------------
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Global pipeline runner (single-user mode for research)
runner = PipelineRunner()
pipeline_lock = threading.Lock()
pipeline_thread = None


def get_output_dir():
    return runner.config.output_dir


def get_models_dir():
    return runner.config.models_dir


# ----------------------------------------------------------------
# Static Frontend
# ----------------------------------------------------------------
@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')


# ----------------------------------------------------------------
# API: Status
# ----------------------------------------------------------------
@app.route('/api/status')
def api_status():
    """Return current pipeline status and available data."""
    status = runner.get_status()
    
    # Check what checkpoints exist
    models_dir = get_models_dir()
    checkpoints = {
        'preprocessed': (models_dir / 'preprocessed_adata.h5ad').exists(),
        'mechanotyped': (models_dir / 'mechanotyped_adata.h5ad').exists(),
        'velocity': (models_dir / 'velocity_adata.h5ad').exists(),
        'final': (models_dir / 'final_adata.h5ad').exists(),
    }
    
    # Check what plots exist
    plots = runner.get_available_plots()
    
    return jsonify({
        'status': status,
        'checkpoints': checkpoints,
        'plots': {k: f'/api/plots/{k}' for k in plots},
        'dataset_loaded': runner.adata is not None,
    })


# ----------------------------------------------------------------
# API: Upload Dataset
# ----------------------------------------------------------------
@app.route('/api/upload', methods=['POST'])
def api_upload():
    """
    Upload dataset files.
    
    Expects multipart form with:
    - h5_file: The filtered_feature_bc_matrix.h5 file
    - spatial_files: Multiple spatial files (images, CSV, JSON)
    """
    if 'h5_file' not in request.files:
        return jsonify({'success': False, 'error': 'No H5 file uploaded'}), 400
    
    h5_file = request.files['h5_file']
    dataset_name = request.form.get('dataset_name', 'uploaded_dataset')
    
    # Create upload directory
    upload_dir = runner.config.data_dir / dataset_name
    spatial_dir = upload_dir / 'spatial'
    upload_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)
    
    # Save H5 file
    h5_path = upload_dir / 'filtered_feature_bc_matrix.h5'
    h5_file.save(str(h5_path))
    
    # Save spatial files
    spatial_files = request.files.getlist('spatial_files')
    saved_files = [h5_path.name]
    for f in spatial_files:
        if f.filename:
            save_path = spatial_dir / os.path.basename(f.filename)
            f.save(str(save_path))
            saved_files.append(f'spatial/{save_path.name}')
    
    # Update config
    runner.config.dataset_name = dataset_name
    
    return jsonify({
        'success': True,
        'dataset_name': dataset_name,
        'files_saved': saved_files,
        'path': str(upload_dir),
    })


@app.route('/api/use-sample', methods=['POST'])
def api_use_sample():
    """Use the built-in sample dataset (V1_Breast_Cancer_Block_A)."""
    sample_path = runner.config.data_dir / 'V1_Breast_Cancer_Block_A'
    
    if not sample_path.exists():
        return jsonify({
            'success': False,
            'error': 'Sample dataset not found. Please download it first.',
        }), 404
    
    runner.config.dataset_name = 'V1_Breast_Cancer_Block_A'
    
    return jsonify({
        'success': True,
        'dataset_name': 'V1_Breast_Cancer_Block_A',
        'path': str(sample_path),
    })


# ----------------------------------------------------------------
# API: Run Pipeline
# ----------------------------------------------------------------
def _run_in_background(func, *args, **kwargs):
    """Run a pipeline stage in a background thread."""
    global pipeline_thread
    
    def wrapper():
        with pipeline_lock:
            func(*args, **kwargs)
    
    pipeline_thread = threading.Thread(target=wrapper, daemon=True)
    pipeline_thread.start()


@app.route('/api/run/full', methods=['POST'])
def api_run_full():
    """Run the complete pipeline (all 4 stages)."""
    if pipeline_lock.locked():
        return jsonify({'success': False, 'error': 'Pipeline already running'}), 409
    
    data = request.get_json(silent=True) or {}
    data_path = data.get('data_path')
    
    _run_in_background(runner.run_full, data_path)
    
    return jsonify({
        'success': True,
        'message': 'Full pipeline started. Poll /api/status for progress.',
    })


@app.route('/api/run/preprocess', methods=['POST'])
def api_run_preprocess():
    """Run Stage 1 only."""
    if pipeline_lock.locked():
        return jsonify({'success': False, 'error': 'Pipeline already running'}), 409
    
    data = request.get_json(silent=True) or {}
    data_path = data.get('data_path')
    
    _run_in_background(runner.run_preprocessing, data_path)
    
    return jsonify({
        'success': True,
        'message': 'Preprocessing started.',
    })


@app.route('/api/run/mechanotype', methods=['POST'])
def api_run_mechanotype():
    """Run Stage 2 only."""
    if pipeline_lock.locked():
        return jsonify({'success': False, 'error': 'Pipeline already running'}), 409
    
    _run_in_background(runner.run_mechanotyping)
    
    return jsonify({
        'success': True,
        'message': 'Mechanotyping started.',
    })


@app.route('/api/run/graph', methods=['POST'])
def api_run_graph():
    """Run Stage 3 only."""
    if pipeline_lock.locked():
        return jsonify({'success': False, 'error': 'Pipeline already running'}), 409
    
    _run_in_background(runner.run_graph_velocity)
    
    return jsonify({
        'success': True,
        'message': 'Graph & velocity computation started.',
    })


@app.route('/api/run/clinical', methods=['POST'])
def api_run_clinical():
    """Run Stage 4 only."""
    if pipeline_lock.locked():
        return jsonify({'success': False, 'error': 'Pipeline already running'}), 409
    
    _run_in_background(runner.run_clinical_scoring)
    
    return jsonify({
        'success': True,
        'message': 'Clinical scoring started.',
    })


# ----------------------------------------------------------------
# API: Drug Simulation
# ----------------------------------------------------------------
@app.route('/api/run/drug-sim', methods=['POST'])
def api_drug_sim():
    """Run drug simulation."""
    if pipeline_lock.locked():
        return jsonify({'success': False, 'error': 'Pipeline already running'}), 409
    
    data = request.get_json(silent=True) or {}
    target_gene = data.get('target_gene', 'LOX')
    reduction_pct = float(data.get('reduction_pct', 100.0))
    
    def run():
        result = runner.run_drug_simulation(target_gene, reduction_pct)
        runner.status['message'] = json.dumps(result, default=str)
    
    _run_in_background(run)
    
    return jsonify({
        'success': True,
        'message': f'Drug simulation started: {target_gene} at {reduction_pct}% reduction',
    })


# ----------------------------------------------------------------
# API: Results
# ----------------------------------------------------------------
@app.route('/api/results')
def api_results():
    """Get results from the latest run."""
    output_dir = get_output_dir()
    
    # Load clinical report if exists
    report_path = output_dir / 'clinical_report.json'
    report = None
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    
    # List available plots
    plots = {}
    if output_dir.exists():
        for f in output_dir.glob('*.png'):
            plots[f.stem] = f'/api/plots/{f.stem}'
    
    return jsonify({
        'report': report,
        'plots': plots,
        'status': runner.get_status(),
    })


@app.route('/api/plots/<name>')
def api_get_plot(name):
    """Serve a generated plot image."""
    output_dir = get_output_dir()
    plot_path = output_dir / f'{name}.png'
    
    if not plot_path.exists():
        return jsonify({'error': f'Plot not found: {name}'}), 404
    
    return send_file(str(plot_path), mimetype='image/png')


# ----------------------------------------------------------------
# API: History
# ----------------------------------------------------------------
@app.route('/api/history')
def api_history():
    """Get past analysis runs from the database."""
    try:
        db = DatabaseManager(config=runner.config)
        runs = db.get_analysis_runs(limit=50)
        
        # Enrich with clinical reports
        for run in runs:
            reports = db.get_clinical_reports(run_id=run['id'])
            run['clinical_report'] = reports[0] if reports else None
        
        return jsonify({'runs': runs})
    except Exception as e:
        return jsonify({'runs': [], 'error': str(e)})


@app.route('/api/history/<int:run_id>')
def api_history_detail(run_id):
    """Get detailed results for a specific run."""
    try:
        db = DatabaseManager(config=runner.config)
        reports = db.get_clinical_reports(run_id=run_id)
        spots = db.get_spot_data(run_id)
        
        return jsonify({
            'report': reports[0] if reports else None,
            'n_spots': len(spots),
            'spot_summary': {
                'n_tumor': sum(1 for s in spots if s.get('is_tumor')),
                'n_tcell': sum(1 for s in spots if s.get('is_tcell')),
                'n_trapped': sum(1 for s in spots if s.get('is_trapped')),
            } if spots else None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------
# API: Config
# ----------------------------------------------------------------
@app.route('/api/config', methods=['GET'])
def api_get_config():
    """Get current configuration."""
    return jsonify(runner.config.to_dict())


@app.route('/api/config', methods=['POST'])
def api_set_config():
    """Update configuration parameters."""
    data = request.get_json(silent=True) or {}
    
    if 'alpha' in data:
        runner.config.mechanotyping.alpha = float(data['alpha'])
    if 'beta' in data:
        runner.config.mechanotyping.beta = float(data['beta'])
    if 'gamma' in data:
        runner.config.mechanotyping.gamma = float(data['gamma'])
    if 'wall_threshold' in data:
        runner.config.mechanotyping.wall_threshold = float(data['wall_threshold'])
    if 'fluid_threshold' in data:
        runner.config.mechanotyping.fluid_threshold = float(data['fluid_threshold'])
    
    return jsonify({
        'success': True,
        'config': runner.config.to_dict(),
    })


# ----------------------------------------------------------------
# API: GNN Model Status & Results
# ----------------------------------------------------------------
@app.route('/api/gnn/status')
def api_gnn_status():
    """Check if GNN weights and results are available."""
    models_dir = get_models_dir()
    output_dir = get_output_dir()
    
    weights_path = models_dir / 'gnn_weights.pt'
    results_path = output_dir / 'gnn_validation_results.json'
    
    return jsonify({
        'weights_available': weights_path.exists(),
        'results_available': results_path.exists(),
        'weights_path': str(weights_path),
    })


@app.route('/api/gnn/results')
def api_gnn_results():
    """Return GNN validation results if available."""
    output_dir = get_output_dir()
    results_path = output_dir / 'gnn_validation_results.json'
    
    if not results_path.exists():
        return jsonify({'error': 'GNN results not found. Train the model first.'}), 404
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Check for GNN comparison plots
    gnn_plots = {}
    for name in ['gnn_comparison', 'gnn_training_curves']:
        p = output_dir / f'{name}.png'
        if p.exists():
            gnn_plots[name] = f'/api/plots/{name}'
    
    return jsonify({
        'results': results,
        'plots': gnn_plots,
    })


# ----------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  MECHANO-VELOCITY API SERVER v0.2")
    print("  http://localhost:5000")
    print("=" * 60)
    
    # Check GNN availability
    gnn_weights = runner.config.models_dir / 'gnn_weights.pt'
    if gnn_weights.exists():
        print(f"  GNN weights: ✅ {gnn_weights}")
    else:
        print(f"  GNN weights: ❌ Not found (graph-diffusion mode)")
    
    print("=" * 60)
    
    # Ensure directories exist
    runner.config.output_dir.mkdir(parents=True, exist_ok=True)
    runner.config.models_dir.mkdir(parents=True, exist_ok=True)
    runner.config.data_dir.mkdir(parents=True, exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=False)

