"""
Preprocessing pipeline for spatial transcriptomics data.

Handles filtering, normalization, and quality control.
"""

import scanpy as sc
import numpy as np
from typing import Optional, Dict, Any
from anndata import AnnData

from .config import Config, default_config


class Preprocessor:
    """
    Preprocessing pipeline for Visium spatial transcriptomics data.
    
    Implements the standard preprocessing steps:
    1. Filter spots with low counts (empty glass)
    2. CPM normalization (target_sum=1e4)
    3. Log1p transformation
    4. Optional: Highly variable genes selection
    5. Optional: PCA and neighborhood graph
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or default_config
        self.qc_metrics: Dict[str, Any] = {}
        
    def run(
        self, 
        adata: AnnData,
        filter_spots: bool = True,
        normalize: bool = True,
        log_transform: bool = True,
        compute_hvg: bool = False,
        compute_pca: bool = False,
        compute_neighbors: bool = False,
        copy: bool = True
    ) -> AnnData:
        """
        Run the full preprocessing pipeline.
        
        Args:
            adata: Input AnnData object.
            filter_spots: Whether to filter low-count spots.
            normalize: Whether to apply CPM normalization.
            log_transform: Whether to apply log1p transformation.
            compute_hvg: Whether to compute highly variable genes.
            compute_pca: Whether to compute PCA.
            compute_neighbors: Whether to compute neighborhood graph.
            copy: Whether to copy the data before modifying.
            
        Returns:
            Preprocessed AnnData object.
        """
        if copy:
            adata = adata.copy()
        
        print("=" * 50)
        print("PREPROCESSING PIPELINE")
        print("=" * 50)
        
        # Store original counts
        n_spots_orig = adata.n_obs
        n_genes_orig = adata.n_vars
        
        # Step 1: Calculate QC metrics
        print("\n1. Calculating QC metrics...")
        self._calculate_qc_metrics(adata)
        
        # Step 2: Filter spots
        if filter_spots:
            print("\n2. Filtering low-quality spots...")
            adata = self._filter_spots(adata)
            print(f"   Kept {adata.n_obs}/{n_spots_orig} spots")
        
        # Step 3: Normalize
        if normalize:
            print("\n3. Normalizing to CPM...")
            self._normalize(adata)
        
        # Step 4: Log transform
        if log_transform:
            print("\n4. Log-transforming...")
            self._log_transform(adata)
        
        # Step 5: Highly variable genes
        if compute_hvg:
            print("\n5. Finding highly variable genes...")
            self._compute_hvg(adata)
        
        # Step 6: PCA
        if compute_pca:
            print("\n6. Computing PCA...")
            self._compute_pca(adata)
        
        # Step 7: Neighbors
        if compute_neighbors:
            print("\n7. Computing neighborhood graph...")
            self._compute_neighbors(adata)
        
        # Update metadata
        adata.uns['preprocessing'] = {
            'filtered': filter_spots,
            'normalized': normalize,
            'log_transformed': log_transform,
            'hvg_computed': compute_hvg,
            'pca_computed': compute_pca,
            'neighbors_computed': compute_neighbors,
            'min_counts': self.config.preprocessing.min_counts,
            'target_sum': self.config.preprocessing.target_sum,
        }
        
        print("\n" + "=" * 50)
        print("Preprocessing complete!")
        print(f"Final shape: {adata.n_obs} spots × {adata.n_vars} genes")
        print("=" * 50)
        
        return adata
    
    def _calculate_qc_metrics(self, adata: AnnData) -> None:
        """Calculate quality control metrics."""
        # Total counts per spot
        adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        
        # Number of genes detected per spot
        adata.obs['n_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten()
        
        # Fraction of mitochondrial genes
        mito_genes = adata.var_names.str.startswith('MT-')
        if mito_genes.sum() > 0:
            adata.obs['pct_mito'] = np.array(
                adata[:, mito_genes].X.sum(axis=1)
            ).flatten() / adata.obs['n_counts'] * 100
        else:
            adata.obs['pct_mito'] = 0.0
        
        # Store summary statistics
        self.qc_metrics = {
            'median_counts': float(np.median(adata.obs['n_counts'])),
            'median_genes': float(np.median(adata.obs['n_genes'])),
            'mean_pct_mito': float(np.mean(adata.obs['pct_mito'])),
        }
        
        print(f"   Median counts/spot: {self.qc_metrics['median_counts']:.0f}")
        print(f"   Median genes/spot: {self.qc_metrics['median_genes']:.0f}")
        
    def _filter_spots(self, adata: AnnData) -> AnnData:
        """Filter spots based on QC thresholds."""
        # Filter by minimum counts
        min_counts = self.config.preprocessing.min_counts
        min_genes = self.config.preprocessing.min_genes
        
        # Create mask
        count_mask = adata.obs['n_counts'] >= min_counts
        gene_mask = adata.obs['n_genes'] >= min_genes
        
        # Remove high mitochondrial spots (likely dying cells)
        mito_mask = adata.obs['pct_mito'] < 20
        
        # Combined mask
        keep_mask = count_mask & gene_mask & mito_mask
        
        print(f"   Removed by low counts (<{min_counts}): {(~count_mask).sum()}")
        print(f"   Removed by low genes (<{min_genes}): {(~gene_mask).sum()}")
        print(f"   Removed by high mito (>20%): {(~mito_mask).sum()}")
        
        return adata[keep_mask].copy()
    
    def _normalize(self, adata: AnnData) -> None:
        """Apply CPM normalization."""
        # Store raw counts before normalization
        adata.layers['counts'] = adata.X.copy()
        
        # Normalize to target sum (counts per million variant)
        sc.pp.normalize_total(
            adata, 
            target_sum=self.config.preprocessing.target_sum
        )
        
    def _log_transform(self, adata: AnnData) -> None:
        """Apply log1p transformation."""
        # Store normalized counts before log transform
        if 'normalized' not in adata.layers:
            adata.layers['normalized'] = adata.X.copy()
        
        # Log transform: log(x + 1)
        sc.pp.log1p(adata)
        
    def _compute_hvg(self, adata: AnnData) -> None:
        """Find highly variable genes."""
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=self.config.preprocessing.n_top_genes,
            flavor='seurat_v3' if 'counts' in adata.layers else 'seurat',
            layer='counts' if 'counts' in adata.layers else None,
        )
        
        n_hvg = adata.var['highly_variable'].sum()
        print(f"   Found {n_hvg} highly variable genes")
        
    def _compute_pca(self, adata: AnnData) -> None:
        """Compute PCA."""
        # Scale before PCA
        sc.pp.scale(adata, max_value=10)
        
        sc.tl.pca(
            adata,
            n_comps=self.config.preprocessing.n_pcs,
            svd_solver='arpack'
        )
        
        print(f"   Computed {self.config.preprocessing.n_pcs} principal components")
        
    def _compute_neighbors(self, adata: AnnData) -> None:
        """Compute neighborhood graph."""
        sc.pp.neighbors(
            adata,
            n_neighbors=self.config.preprocessing.n_neighbors,
            n_pcs=self.config.preprocessing.n_pcs
        )
        
        print(f"   Built graph with {self.config.preprocessing.n_neighbors} neighbors")


def preprocess_visium(
    adata: AnnData,
    config: Optional[Config] = None,
    **kwargs
) -> AnnData:
    """
    Convenience function for preprocessing.
    
    Args:
        adata: Input AnnData object.
        config: Configuration object.
        **kwargs: Additional arguments passed to Preprocessor.run().
        
    Returns:
        Preprocessed AnnData object.
    """
    preprocessor = Preprocessor(config)
    return preprocessor.run(adata, **kwargs)
