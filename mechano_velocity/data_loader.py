"""
Data loading utilities for 10x Genomics Visium spatial transcriptomics data.

Handles loading, validation, and initial data structure creation.
"""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from anndata import AnnData

from .config import Config, default_config


class DataLoader:
    """
    Loader for 10x Genomics Visium spatial transcriptomics data.
    
    Handles the complete data ingestion pipeline including:
    - Loading the H5 feature matrix
    - Loading spatial coordinates and images
    - Extracting gene panels for mechanotyping
    - Basic data validation
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or default_config
        self.adata: Optional[AnnData] = None
        
    def load_visium(
        self, 
        path: Optional[Path] = None,
        load_images: bool = True
    ) -> AnnData:
        """
        Load 10x Visium data from Space Ranger output.
        
        Args:
            path: Path to dataset folder. Uses config default if not provided.
            load_images: Whether to load the high-resolution tissue image.
            
        Returns:
            AnnData object with spatial data loaded.
            
        Raises:
            FileNotFoundError: If required files are missing.
        """
        if path is None:
            path = self.config.dataset_path
        path = Path(path)
        
        # Validate required files exist
        if not self._validate_files(path):
            raise FileNotFoundError(
                f"Missing required files in {path}. "
                "Please ensure you have downloaded the complete 10x Visium dataset."
            )
        
        print(f"Loading Visium data from: {path}")
        
        # Use scanpy's read_visium function
        self.adata = sc.read_visium(
            path,
            count_file="filtered_feature_bc_matrix.h5",
            load_images=load_images,
        )
        
        # Add metadata
        self.adata.uns['mechano_velocity'] = {
            'version': '0.1.0',
            'dataset_name': self.config.dataset_name,
            'loaded_from': str(path),
        }
        
        # Make gene names unique (some datasets have duplicates)
        self.adata.var_names_make_unique()
        
        print(f"Loaded {self.adata.n_obs} spots × {self.adata.n_vars} genes")
        
        return self.adata
    
    def _validate_files(self, path: Path) -> bool:
        """
        Validate that all required files exist.
        
        Args:
            path: Path to dataset folder.
            
        Returns:
            True if all files exist, False otherwise.
        """
        required = [
            path / "filtered_feature_bc_matrix.h5",
            path / "spatial" / "tissue_positions_list.csv",
            path / "spatial" / "scalefactors_json.json",
        ]
        
        # Some datasets use tissue_positions.csv instead
        alt_positions = path / "spatial" / "tissue_positions.csv"
        
        missing = []
        for f in required:
            if not f.exists():
                # Check alternative for positions file
                if "tissue_positions_list" in str(f) and alt_positions.exists():
                    continue
                missing.append(f)
        
        if missing:
            print(f"Missing files:")
            for f in missing:
                print(f"  - {f}")
            return False
        return True
    
    def extract_gene_panel(
        self, 
        genes: Optional[List[str]] = None,
        allow_missing: bool = True
    ) -> Tuple[AnnData, List[str]]:
        """
        Extract a subset of genes for mechanotyping analysis.
        
        Args:
            genes: List of gene names. Uses config default if not provided.
            allow_missing: If True, continue with available genes. If False, raise error.
            
        Returns:
            Tuple of (subset AnnData, list of missing genes).
            
        Raises:
            ValueError: If adata not loaded or critical genes missing.
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_visium() first.")
        
        if genes is None:
            genes = self.config.genes.all_genes
        
        # Find available genes
        available = [g for g in genes if g in self.adata.var_names]
        missing = [g for g in genes if g not in self.adata.var_names]
        
        if missing:
            print(f"Warning: Missing genes in dataset: {missing}")
            
            # Check for critical genes (must have at least one collagen)
            critical_missing = [
                g for g in self.config.genes.collagen_genes 
                if g in missing
            ]
            if len(critical_missing) == len(self.config.genes.collagen_genes):
                raise ValueError(
                    "No collagen genes found in dataset. "
                    f"Expected at least one of: {self.config.genes.collagen_genes}"
                )
        
        if not allow_missing and missing:
            raise ValueError(f"Missing required genes: {missing}")
        
        # Create subset
        subset = self.adata[:, available].copy()
        print(f"Extracted {len(available)} genes for mechanotyping")
        
        return subset, missing
    
    def get_spatial_coordinates(self) -> np.ndarray:
        """
        Get spatial coordinates of all spots.
        
        Returns:
            Array of shape (n_spots, 2) with (x, y) coordinates.
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_visium() first.")
        
        return self.adata.obsm['spatial']
    
    def get_spot_data(self) -> pd.DataFrame:
        """
        Get comprehensive spot-level metadata.
        
        Returns:
            DataFrame with spot coordinates, array positions, and QC metrics.
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_visium() first.")
        
        coords = self.get_spatial_coordinates()
        
        df = pd.DataFrame({
            'spot_id': self.adata.obs_names,
            'x': coords[:, 0],
            'y': coords[:, 1],
            'n_counts': np.array(self.adata.X.sum(axis=1)).flatten(),
            'n_genes': np.array((self.adata.X > 0).sum(axis=1)).flatten(),
        })
        
        # Add array row/col if available
        if 'array_row' in self.adata.obs.columns:
            df['array_row'] = self.adata.obs['array_row'].values
            df['array_col'] = self.adata.obs['array_col'].values
        
        return df
    
    def summary(self) -> str:
        """
        Generate a summary of the loaded data.
        
        Returns:
            Human-readable summary string.
        """
        if self.adata is None:
            return "No data loaded."
        
        lines = [
            "=" * 50,
            "MECHANO-VELOCITY DATA SUMMARY",
            "=" * 50,
            f"Dataset: {self.config.dataset_name}",
            f"Spots: {self.adata.n_obs:,}",
            f"Genes: {self.adata.n_vars:,}",
            "",
            "Spatial Info:",
            f"  Resolution: {self.config.cell_physics.spot_diameter_um}µm per spot",
            f"  Grid type: Hexagonal ({self.config.cell_physics.neighbors_per_spot} neighbors)",
            "",
            "Gene Panel Availability:",
        ]
        
        # Check gene availability
        for category, genes in [
            ("Collagen", self.config.genes.collagen_genes),
            ("Crosslinkers", self.config.genes.crosslinker_genes),
            ("Degradation", self.config.genes.degradation_genes),
            ("T-cell markers", self.config.genes.tcell_markers),
            ("Tumor markers", self.config.genes.tumor_markers),
        ]:
            available = sum(1 for g in genes if g in self.adata.var_names)
            lines.append(f"  {category}: {available}/{len(genes)} available")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def load_visium_data(
    path: Optional[Path] = None,
    config: Optional[Config] = None
) -> AnnData:
    """
    Convenience function to load Visium data.
    
    Args:
        path: Path to dataset folder.
        config: Configuration object.
        
    Returns:
        Loaded AnnData object.
    """
    loader = DataLoader(config)
    return loader.load_visium(path)
