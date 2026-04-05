"""
Mechano-Velocity Pipeline Runner

Orchestrates all 4 stages of the analysis pipeline.
Can be called from the API server or used as a standalone CLI tool.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import scanpy as sc
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from mechano_velocity import (
    Config, DataLoader, Preprocessor, Mechanotyper,
    GraphBuilder, VelocityCorrector, ClinicalScorer,
    Visualizer, DatabaseManager
)


class PipelineRunner:
    """
    End-to-end pipeline runner for Mechano-Velocity analysis.
    
    Chains all 4 stages and persists results to disk and database.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.adata = None
        self.run_id = None
        self.db = None
        self.status = {
            'current_stage': None,
            'stages_completed': [],
            'progress': 0,
            'message': 'Idle',
            'error': None,
        }
        self._logs = []
    
    def _log(self, message: str):
        """Add a log entry."""
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self._logs.append(entry)
        print(entry)
    
    def _update_status(self, stage: str, progress: int, message: str):
        """Update pipeline status."""
        self.status['current_stage'] = stage
        self.status['progress'] = progress
        self.status['message'] = message
    
    def get_status(self) -> Dict[str, Any]:
        """Return current pipeline status."""
        return {**self.status, 'logs': self._logs[-50:]}  # Last 50 log entries
    
    # ----------------------------------------------------------------
    # Stage 1: Preprocessing
    # ----------------------------------------------------------------
    def run_preprocessing(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run Stage 1: Load and preprocess spatial transcriptomics data.
        
        Args:
            data_path: Path to dataset directory. Uses config default if None.
        """
        self._update_status('preprocessing', 10, 'Loading data...')
        self._log("=" * 50)
        self._log("STAGE 1: PREPROCESSING")
        self._log("=" * 50)
        
        try:
            # Initialize database
            self.db = DatabaseManager(config=self.config)
            
            # Load data
            if data_path:
                self.config.data_dir = Path(data_path).parent
                self.config.dataset_name = Path(data_path).name
            
            self._log(f"Loading dataset: {self.config.dataset_name}")
            loader = DataLoader(self.config)
            self.adata = loader.load_visium()
            
            self._log(f"Loaded: {self.adata.shape[0]} spots × {self.adata.shape[1]} genes")
            
            # Start DB run
            self.run_id = self.db.start_analysis_run(
                sample_id=self.config.dataset_name,
                n_spots=self.adata.n_obs,
                n_genes=self.adata.n_vars,
                config_dict=self.config.to_dict()
            )
            
            # Preprocess
            self._update_status('preprocessing', 30, 'Running QC and normalization...')
            preprocessor = Preprocessor(self.config)
            self.adata = preprocessor.run(self.adata)
            
            self._log(f"After preprocessing: {self.adata.shape[0]} spots × {self.adata.shape[1]} genes")
            
            # Save checkpoint
            self._update_status('preprocessing', 45, 'Saving checkpoint...')
            checkpoint_path = self.config.models_dir / 'preprocessed_adata.h5ad'
            self.adata.write_h5ad(checkpoint_path)
            self._log(f"Saved checkpoint: {checkpoint_path}")
            
            self.status['stages_completed'].append('preprocessing')
            self._update_status('preprocessing', 50, 'Preprocessing complete')
            
            return {
                'success': True,
                'n_spots': self.adata.n_obs,
                'n_genes': self.adata.n_vars,
                'clusters': int(self.adata.obs['leiden'].nunique()) if 'leiden' in self.adata.obs.columns else 0,
                'checkpoint': str(checkpoint_path),
            }
            
        except Exception as e:
            self.status['error'] = str(e)
            self._log(f"ERROR in preprocessing: {e}")
            return {'success': False, 'error': str(e)}
    
    # ----------------------------------------------------------------
    # Stage 2: Mechanotyping
    # ----------------------------------------------------------------
    def run_mechanotyping(self) -> Dict[str, Any]:
        """Run Stage 2: Calculate ECM resistance field."""
        self._update_status('mechanotyping', 55, 'Computing resistance field...')
        self._log("=" * 50)
        self._log("STAGE 2: MECHANOTYPING")
        self._log("=" * 50)
        
        try:
            if self.adata is None:
                self._load_checkpoint('preprocessed_adata.h5ad')
            
            # Calculate resistance
            mechanotyper = Mechanotyper(self.config)
            resistance = mechanotyper.calculate_resistance(self.adata)
            
            self._log(f"Resistance range: [{resistance.min():.3f}, {resistance.max():.3f}]")
            
            # Count categories
            categories = self.adata.obs['resistance_category'].value_counts().to_dict()
            self._log(f"Categories: {categories}")
            
            # Generate resistance heatmap
            self._update_status('mechanotyping', 65, 'Generating resistance plots...')
            viz = Visualizer(self.config)
            
            plot_path = self.config.output_dir / 'resistance_map.png'
            viz.plot_resistance_heatmap(self.adata, show_image=False, save_path=str(plot_path))
            plt.close('all')
            
            # Drug simulation
            self._update_status('mechanotyping', 70, 'Running drug simulation...')
            original_resistance = self.adata.obs['resistance'].values.copy()
            sim_resistance = mechanotyper.simulate_drug(self.adata, target_gene='LOX', reduction_factor=1.0)
            
            drug_plot_path = self.config.output_dir / 'drug_simulation.png'
            viz.plot_drug_simulation(
                self.adata, original_resistance, sim_resistance,
                drug_name='LOX Inhibitor', save_path=str(drug_plot_path)
            )
            plt.close('all')
            
            # Save checkpoint
            checkpoint_path = self.config.models_dir / 'mechanotyped_adata.h5ad'
            self.adata.write_h5ad(checkpoint_path)
            self._log(f"Saved checkpoint: {checkpoint_path}")
            
            self.status['stages_completed'].append('mechanotyping')
            self._update_status('mechanotyping', 75, 'Mechanotyping complete')
            
            return {
                'success': True,
                'resistance_mean': float(resistance.mean()),
                'resistance_std': float(resistance.std()),
                'categories': {k: int(v) for k, v in categories.items()},
                'plots': {
                    'resistance_map': str(plot_path),
                    'drug_simulation': str(drug_plot_path),
                },
                'checkpoint': str(checkpoint_path),
            }
            
        except Exception as e:
            self.status['error'] = str(e)
            self._log(f"ERROR in mechanotyping: {e}")
            return {'success': False, 'error': str(e)}
    
    # ----------------------------------------------------------------
    # Stage 3: Graph Construction & Velocity Correction
    # ----------------------------------------------------------------
    def run_graph_velocity(self) -> Dict[str, Any]:
        """Run Stage 3: Build spatial graph and compute corrected velocity."""
        self._update_status('graph_velocity', 78, 'Building spatial graph...')
        self._log("=" * 50)
        self._log("STAGE 3: GRAPH & VELOCITY")
        self._log("=" * 50)
        
        try:
            if self.adata is None:
                self._load_checkpoint('mechanotyped_adata.h5ad')
            
            # Build spatial graph
            graph_builder = GraphBuilder(self.config)
            adjacency = graph_builder.build_spatial_graph(
                self.adata,
                method='knn',
                k_neighbors=6,
                include_resistance=True,
                include_similarity=True
            )
            
            graph_metrics = graph_builder.metrics.to_dict()
            self._log(f"Graph: {graph_metrics['n_nodes']} nodes, {graph_metrics['n_edges']} edges")
            
            # Compute velocity
            self._update_status('graph_velocity', 83, 'Computing corrected velocity...')
            velocity_corrector = VelocityCorrector(self.config)
            corrected = velocity_corrector.apply_resistance_correction(
                self.adata,
                graph_builder=graph_builder,
                method='projection'
            )
            
            self._log(f"Velocity shape: {corrected.shape}")
            self._log(f"Mean magnitude: {np.linalg.norm(corrected, axis=1).mean():.4f}")
            
            # Identify trapped cells
            self._update_status('graph_velocity', 86, 'Identifying trapped cells...')
            trapped = velocity_corrector.identify_trapped_cells(
                self.adata,
                velocity_threshold=0.01,
                resistance_threshold=self.config.mechanotyping.wall_threshold
            )
            self._log(f"Trapped cells: {trapped.sum()}")
            
            # Generate plots
            self._update_status('graph_velocity', 88, 'Generating velocity plots...')
            viz = Visualizer(self.config)
            
            arrows_path = self.config.output_dir / 'velocity_arrows.png'
            viz.plot_velocity_arrows(
                self.adata, velocity_key='velocity_corrected',
                color_by='resistance', save_path=str(arrows_path)
            )
            plt.close('all')
            
            stream_path = self.config.output_dir / 'velocity_streamplot.png'
            viz.plot_velocity_streamplot(
                self.adata, velocity_key='velocity_corrected',
                color_by='resistance', save_path=str(stream_path)
            )
            plt.close('all')
            
            overview_path = self.config.output_dir / 'analysis_overview.png'
            viz.plot_comparison(self.adata, save_path=str(overview_path))
            plt.close('all')
            
            # Save checkpoint
            checkpoint_path = self.config.models_dir / 'velocity_adata.h5ad'
            self.adata.write_h5ad(checkpoint_path)
            self._log(f"Saved checkpoint: {checkpoint_path}")
            
            self.status['stages_completed'].append('graph_velocity')
            self._update_status('graph_velocity', 90, 'Graph & velocity complete')
            
            return {
                'success': True,
                'graph': graph_metrics,
                'trapped_cells': int(trapped.sum()),
                'mean_velocity': float(np.linalg.norm(corrected, axis=1).mean()),
                'plots': {
                    'velocity_arrows': str(arrows_path),
                    'velocity_streamplot': str(stream_path),
                    'analysis_overview': str(overview_path),
                },
                'checkpoint': str(checkpoint_path),
            }
            
        except Exception as e:
            self.status['error'] = str(e)
            self._log(f"ERROR in graph/velocity: {e}")
            return {'success': False, 'error': str(e)}
    
    # ----------------------------------------------------------------
    # Stage 4: Clinical Scoring
    # ----------------------------------------------------------------
    def run_clinical_scoring(self) -> Dict[str, Any]:
        """Run Stage 4: Generate clinical report and scores."""
        self._update_status('clinical', 92, 'Computing clinical scores...')
        self._log("=" * 50)
        self._log("STAGE 4: CLINICAL SCORING")
        self._log("=" * 50)
        
        try:
            if self.adata is None:
                self._load_checkpoint('velocity_adata.h5ad')
            
            # Generate clinical report
            scorer = ClinicalScorer(self.config)
            report = scorer.generate_report(
                self.adata,
                sample_id=self.config.dataset_name
            )
            
            self._log(f"MTS Score: {report.mechano_therapeutic_score:.4f}")
            self._log(f"Risk Category: {report.risk_category}")
            self._log(f"Recommendation: {report.therapeutic_recommendation}")
            
            # Save report
            self._update_status('clinical', 95, 'Saving clinical report...')
            report_txt_path = self.config.output_dir / 'clinical_report.txt'
            scorer.save_report(str(report_txt_path), format='txt')
            
            report_json_path = self.config.output_dir / 'clinical_report.json'
            scorer.save_report(str(report_json_path), format='json')
            
            # Save to database
            if self.db and self.run_id:
                self.db.save_clinical_report(self.run_id, report.to_dict())
                self.db.save_spot_data(self.run_id, self.adata)
                self.db.complete_analysis_run(self.run_id, status='completed')
            
            # Save final checkpoint
            final_path = self.config.models_dir / 'final_adata.h5ad'
            self.adata.write_h5ad(final_path)
            self._log(f"Saved final model: {final_path}")
            
            # Save config
            self.config.save(self.config.models_dir / 'config.json')
            
            self.status['stages_completed'].append('clinical')
            self._update_status('clinical', 100, 'Pipeline complete!')
            
            return {
                'success': True,
                'report': report.to_dict(),
                'report_text': report.to_text(),
                'files': {
                    'report_txt': str(report_txt_path),
                    'report_json': str(report_json_path),
                    'final_model': str(final_path),
                },
            }
            
        except Exception as e:
            self.status['error'] = str(e)
            self._log(f"ERROR in clinical scoring: {e}")
            return {'success': False, 'error': str(e)}
    
    # ----------------------------------------------------------------
    # Full Pipeline
    # ----------------------------------------------------------------
    def run_full(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete 4-stage pipeline.
        
        Args:
            data_path: Path to dataset directory.
            
        Returns:
            Dictionary with results from all stages.
        """
        self._logs = []
        self.status = {
            'current_stage': None,
            'stages_completed': [],
            'progress': 0,
            'message': 'Starting full pipeline...',
            'error': None,
        }
        
        results = {}
        
        # Stage 1
        result = self.run_preprocessing(data_path)
        results['preprocessing'] = result
        if not result.get('success'):
            return results
        
        # Stage 2
        result = self.run_mechanotyping()
        results['mechanotyping'] = result
        if not result.get('success'):
            return results
        
        # Stage 3
        result = self.run_graph_velocity()
        results['graph_velocity'] = result
        if not result.get('success'):
            return results
        
        # Stage 4
        result = self.run_clinical_scoring()
        results['clinical'] = result
        
        return results
    
    # ----------------------------------------------------------------
    # Drug Simulation
    # ----------------------------------------------------------------
    def run_drug_simulation(
        self, 
        target_gene: str = 'LOX',
        reduction_pct: float = 100.0
    ) -> Dict[str, Any]:
        """
        Run drug simulation on current data.
        
        Args:
            target_gene: Gene to target (LOX, MMP9, COL1A1, etc.)
            reduction_pct: Reduction percentage (0-100).
        """
        try:
            if self.adata is None:
                self._load_checkpoint('mechanotyped_adata.h5ad')
            
            mechanotyper = Mechanotyper(self.config)
            
            # Ensure resistance is calculated
            if 'resistance' not in self.adata.obs.columns:
                mechanotyper.calculate_resistance(self.adata)
            
            original = self.adata.obs['resistance'].values.copy()
            reduction_factor = reduction_pct / 100.0
            simulated = mechanotyper.simulate_drug(
                self.adata, target_gene=target_gene, reduction_factor=reduction_factor
            )
            
            # Generate plot
            viz = Visualizer(self.config)
            drug_name = f"{target_gene} Inhibitor ({reduction_pct:.0f}%)"
            plot_path = self.config.output_dir / f'drug_sim_{target_gene}_{int(reduction_pct)}.png'
            viz.plot_drug_simulation(
                self.adata, original, simulated,
                drug_name=drug_name, save_path=str(plot_path)
            )
            plt.close('all')
            
            return {
                'success': True,
                'target_gene': target_gene,
                'reduction_pct': reduction_pct,
                'original_mean': float(original.mean()),
                'simulated_mean': float(simulated.mean()),
                'delta_mean': float(simulated.mean() - original.mean()),
                'plot': str(plot_path),
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _load_checkpoint(self, filename: str):
        """Load the most recent checkpoint."""
        path = self.config.models_dir / filename
        if not path.exists():
            # Try output dir as fallback
            path = self.config.output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filename}")
        
        self._log(f"Loading checkpoint: {path}")
        self.adata = sc.read_h5ad(path)
        
        # Re-initialize DB if needed
        if self.db is None:
            self.db = DatabaseManager(config=self.config)
    
    def get_available_plots(self) -> Dict[str, str]:
        """List all generated plot files."""
        plots = {}
        if self.config.output_dir.exists():
            for f in self.config.output_dir.glob('*.png'):
                plots[f.stem] = str(f)
        return plots


# CLI entry point
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Mechano-Velocity Pipeline Runner')
    parser.add_argument('--data', type=str, help='Path to dataset directory')
    parser.add_argument('--stage', type=str, default='full',
                       choices=['full', 'preprocess', 'mechanotype', 'graph', 'clinical'],
                       help='Which stage to run')
    args = parser.parse_args()
    
    runner = PipelineRunner()
    
    if args.stage == 'full':
        results = runner.run_full(args.data)
    elif args.stage == 'preprocess':
        results = runner.run_preprocessing(args.data)
    elif args.stage == 'mechanotype':
        results = runner.run_mechanotyping()
    elif args.stage == 'graph':
        results = runner.run_graph_velocity()
    elif args.stage == 'clinical':
        results = runner.run_clinical_scoring()
    
    print("\n" + json.dumps(results, indent=2, default=str))
