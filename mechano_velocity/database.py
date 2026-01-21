"""
Database management for storing Mechano-Velocity outputs.

Provides persistent storage for:
- Analysis runs and configurations
- Clinical reports
- Model outputs for validation
- Comparison across samples
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import asdict
import numpy as np

from .config import Config, default_config


class DatabaseManager:
    """
    SQLite database for storing analysis outputs.
    
    Tables:
    - analysis_runs: Track each analysis execution
    - clinical_reports: Store clinical scoring results
    - model_outputs: Store detailed model predictions
    - validation_logs: Track validation results
    """
    
    def __init__(
        self, 
        db_path: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. Uses config default if None.
            config: Configuration object.
        """
        self.config = config or default_config
        
        if db_path is None:
            db_path = self.config.output_dir / self.config.database.db_path
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Analysis runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT NOT NULL,
                    run_timestamp TEXT NOT NULL,
                    config_json TEXT,
                    n_spots INTEGER,
                    n_genes INTEGER,
                    status TEXT DEFAULT 'started',
                    notes TEXT
                )
            ''')
            
            # Clinical reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clinical_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    sample_id TEXT NOT NULL,
                    report_timestamp TEXT NOT NULL,
                    metastatic_risk_score REAL,
                    immune_exclusion_score REAL,
                    mts_score REAL,
                    risk_category TEXT,
                    recommendation TEXT,
                    n_tumor_spots INTEGER,
                    n_tcell_spots INTEGER,
                    n_boundary_spots INTEGER,
                    mean_boundary_resistance REAL,
                    details_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            ''')
            
            # Model outputs table (for detailed predictions)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    output_type TEXT NOT NULL,
                    data_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            ''')
            
            # Validation logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    validation_type TEXT NOT NULL,
                    expected_value TEXT,
                    actual_value TEXT,
                    passed INTEGER,
                    notes TEXT,
                    validated_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            ''')
            
            # Spot-level data table (for detailed analysis)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spot_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    spot_id TEXT NOT NULL,
                    x_coord REAL,
                    y_coord REAL,
                    resistance REAL,
                    velocity_magnitude REAL,
                    velocity_x REAL,
                    velocity_y REAL,
                    is_tumor INTEGER,
                    is_tcell INTEGER,
                    is_boundary INTEGER,
                    is_trapped INTEGER,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            ''')
            
            conn.commit()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def start_analysis_run(
        self,
        sample_id: str,
        n_spots: int,
        n_genes: int,
        config_dict: Optional[Dict] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Record the start of an analysis run.
        
        Args:
            sample_id: Sample identifier.
            n_spots: Number of spots.
            n_genes: Number of genes.
            config_dict: Configuration dictionary.
            notes: Additional notes.
            
        Returns:
            Run ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_runs 
                (sample_id, run_timestamp, config_json, n_spots, n_genes, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                sample_id,
                datetime.now().isoformat(),
                json.dumps(config_dict) if config_dict else None,
                n_spots,
                n_genes,
                'started',
                notes
            ))
            
            conn.commit()
            run_id = cursor.lastrowid
            
        print(f"Started analysis run {run_id} for sample: {sample_id}")
        return run_id
    
    def complete_analysis_run(self, run_id: int, status: str = 'completed') -> None:
        """
        Mark analysis run as completed.
        
        Args:
            run_id: Run ID.
            status: Final status ('completed', 'failed', etc.).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE analysis_runs SET status = ? WHERE id = ?',
                (status, run_id)
            )
            conn.commit()
    
    def save_clinical_report(
        self,
        run_id: int,
        report_dict: Dict[str, Any]
    ) -> int:
        """
        Save clinical report to database.
        
        Args:
            run_id: Associated analysis run ID.
            report_dict: Report dictionary from ClinicalReport.to_dict().
            
        Returns:
            Report ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            scores = report_dict.get('scores', {})
            classification = report_dict.get('classification', {})
            cell_counts = report_dict.get('cell_counts', {})
            
            cursor.execute('''
                INSERT INTO clinical_reports
                (run_id, sample_id, report_timestamp, metastatic_risk_score,
                 immune_exclusion_score, mts_score, risk_category, recommendation,
                 n_tumor_spots, n_tcell_spots, n_boundary_spots, 
                 mean_boundary_resistance, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                report_dict.get('sample_id', 'unknown'),
                report_dict.get('analysis_date', datetime.now().isoformat()),
                scores.get('metastatic_risk'),
                scores.get('immune_exclusion'),
                scores.get('mts'),
                classification.get('risk_category'),
                classification.get('recommendation'),
                cell_counts.get('tumor_spots'),
                cell_counts.get('tcell_spots'),
                cell_counts.get('boundary_spots'),
                report_dict.get('mean_boundary_resistance'),
                json.dumps(report_dict.get('details', {}))
            ))
            
            conn.commit()
            report_id = cursor.lastrowid
            
        print(f"Saved clinical report {report_id}")
        return report_id
    
    def save_spot_data(
        self,
        run_id: int,
        adata: 'AnnData'
    ) -> int:
        """
        Save spot-level data to database.
        
        Args:
            run_id: Analysis run ID.
            adata: AnnData with computed values.
            
        Returns:
            Number of spots saved.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            coords = adata.obsm['spatial']
            n_spots = adata.n_obs
            
            # Get computed values with defaults
            resistance = adata.obs.get('resistance', np.zeros(n_spots))
            velocity_mag = adata.obs.get('velocity_magnitude', np.zeros(n_spots))
            
            velocity = adata.obsm.get('velocity_corrected', np.zeros((n_spots, 2)))
            
            is_tumor = adata.obs.get('is_tumor', np.zeros(n_spots, dtype=bool))
            is_tcell = adata.obs.get('is_tcell', np.zeros(n_spots, dtype=bool))
            is_boundary = adata.obs.get('is_boundary', np.zeros(n_spots, dtype=bool))
            is_trapped = adata.obs.get('is_trapped', np.zeros(n_spots, dtype=bool))
            
            # Batch insert
            data = [
                (
                    run_id,
                    str(adata.obs_names[i]),
                    float(coords[i, 0]),
                    float(coords[i, 1]),
                    float(resistance.iloc[i] if hasattr(resistance, 'iloc') else resistance[i]),
                    float(velocity_mag.iloc[i] if hasattr(velocity_mag, 'iloc') else velocity_mag[i]),
                    float(velocity[i, 0]),
                    float(velocity[i, 1]),
                    int(is_tumor.iloc[i] if hasattr(is_tumor, 'iloc') else is_tumor[i]),
                    int(is_tcell.iloc[i] if hasattr(is_tcell, 'iloc') else is_tcell[i]),
                    int(is_boundary.iloc[i] if hasattr(is_boundary, 'iloc') else is_boundary[i]),
                    int(is_trapped.iloc[i] if hasattr(is_trapped, 'iloc') else is_trapped[i]),
                )
                for i in range(n_spots)
            ]
            
            cursor.executemany('''
                INSERT INTO spot_data
                (run_id, spot_id, x_coord, y_coord, resistance, velocity_magnitude,
                 velocity_x, velocity_y, is_tumor, is_tcell, is_boundary, is_trapped)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            
            conn.commit()
            
        print(f"Saved {n_spots} spots to database")
        return n_spots
    
    def add_validation_log(
        self,
        run_id: int,
        validation_type: str,
        expected: Any,
        actual: Any,
        passed: bool,
        notes: Optional[str] = None
    ) -> None:
        """
        Add validation result to database.
        
        Args:
            run_id: Analysis run ID.
            validation_type: Type of validation (e.g., 'histology_overlay').
            expected: Expected value.
            actual: Actual value.
            passed: Whether validation passed.
            notes: Additional notes.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO validation_logs
                (run_id, validation_type, expected_value, actual_value, passed, notes, validated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                validation_type,
                str(expected),
                str(actual),
                int(passed),
                notes,
                datetime.now().isoformat()
            ))
            
            conn.commit()
    
    def get_analysis_runs(
        self,
        sample_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get analysis runs.
        
        Args:
            sample_id: Filter by sample ID.
            limit: Maximum number of results.
            
        Returns:
            List of run dictionaries.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if sample_id:
                cursor.execute(
                    'SELECT * FROM analysis_runs WHERE sample_id = ? ORDER BY run_timestamp DESC LIMIT ?',
                    (sample_id, limit)
                )
            else:
                cursor.execute(
                    'SELECT * FROM analysis_runs ORDER BY run_timestamp DESC LIMIT ?',
                    (limit,)
                )
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
        return [dict(zip(columns, row)) for row in rows]
    
    def get_clinical_reports(
        self,
        sample_id: Optional[str] = None,
        run_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Get clinical reports.
        
        Args:
            sample_id: Filter by sample ID.
            run_id: Filter by run ID.
            
        Returns:
            List of report dictionaries.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if run_id:
                cursor.execute('SELECT * FROM clinical_reports WHERE run_id = ?', (run_id,))
            elif sample_id:
                cursor.execute('SELECT * FROM clinical_reports WHERE sample_id = ?', (sample_id,))
            else:
                cursor.execute('SELECT * FROM clinical_reports ORDER BY report_timestamp DESC')
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
        return [dict(zip(columns, row)) for row in rows]
    
    def get_spot_data(
        self,
        run_id: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Get spot-level data for a run.
        
        Args:
            run_id: Analysis run ID.
            filters: Optional filters (e.g., {'is_trapped': 1}).
            
        Returns:
            List of spot dictionaries.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM spot_data WHERE run_id = ?'
            params = [run_id]
            
            if filters:
                for key, value in filters.items():
                    query += f' AND {key} = ?'
                    params.append(value)
            
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
        return [dict(zip(columns, row)) for row in rows]
    
    def compare_runs(
        self,
        run_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Compare multiple analysis runs.
        
        Args:
            run_ids: List of run IDs to compare.
            
        Returns:
            Comparison dictionary with summary statistics.
        """
        reports = []
        for run_id in run_ids:
            run_reports = self.get_clinical_reports(run_id=run_id)
            if run_reports:
                reports.append(run_reports[0])
        
        if not reports:
            return {'error': 'No reports found for given run IDs'}
        
        # Extract scores for comparison
        mts_scores = [r['mts_score'] for r in reports if r['mts_score']]
        m_risk_scores = [r['metastatic_risk_score'] for r in reports if r['metastatic_risk_score']]
        i_excl_scores = [r['immune_exclusion_score'] for r in reports if r['immune_exclusion_score']]
        
        return {
            'n_runs': len(reports),
            'run_ids': run_ids,
            'mts': {
                'mean': np.mean(mts_scores) if mts_scores else None,
                'std': np.std(mts_scores) if mts_scores else None,
                'min': min(mts_scores) if mts_scores else None,
                'max': max(mts_scores) if mts_scores else None,
            },
            'metastatic_risk': {
                'mean': np.mean(m_risk_scores) if m_risk_scores else None,
                'std': np.std(m_risk_scores) if m_risk_scores else None,
            },
            'immune_exclusion': {
                'mean': np.mean(i_excl_scores) if i_excl_scores else None,
                'std': np.std(i_excl_scores) if i_excl_scores else None,
            },
            'categories': {r['sample_id']: r['risk_category'] for r in reports},
        }
    
    def export_to_csv(
        self,
        run_id: int,
        output_path: str
    ) -> None:
        """
        Export spot data to CSV.
        
        Args:
            run_id: Analysis run ID.
            output_path: Output CSV path.
        """
        import csv
        
        spots = self.get_spot_data(run_id)
        if not spots:
            print(f"No data found for run {run_id}")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=spots[0].keys())
            writer.writeheader()
            writer.writerows(spots)
        
        print(f"Exported {len(spots)} spots to: {output_path}")
