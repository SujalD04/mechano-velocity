"""
Mechanotyping module: Core resistance field calculation.

This is the unique contribution of Mechano-Velocity - calculating the
physical "Wall Strength" at every spatial spot based on gene signatures.
"""

import numpy as np
from scipy import sparse
from scipy.special import expit  # Sigmoid function
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, Dict
from anndata import AnnData

from .config import Config, default_config


class Mechanotyper:
    """
    Calculate the ECM Resistance Field from gene expression.
    
    The core equation:
    D_i = (α * COL1A1 + α * COL1A2) × (1 + β * LOX) - (γ * MMP9)
    R_i = sigmoid(D_i - μ)
    
    Where:
    - D_i: Raw density score for spot i
    - R_i: Normalized resistance probability [0, 1]
    - α, β, γ: Hyperparameters (weights)
    - μ: Centering parameter (dataset mean)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the mechanotyper.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or default_config
        self.params = self.config.mechanotyping
        self.genes = self.config.genes
        
        # Store intermediate results
        self.raw_density: Optional[np.ndarray] = None
        self.resistance: Optional[np.ndarray] = None
        self.gene_contributions: Dict[str, np.ndarray] = {}
        
    def calculate_resistance(
        self,
        adata: AnnData,
        smooth: bool = True,
        store_in_adata: bool = True
    ) -> np.ndarray:
        """
        Calculate the resistance field for all spots.
        
        Args:
            adata: AnnData object with gene expression.
            smooth: Whether to apply KNN smoothing for zero-inflation.
            store_in_adata: Whether to store results in adata.obs.
            
        Returns:
            Array of resistance values [0, 1] for each spot.
        """
        print("=" * 50)
        print("MECHANOTYPING: Calculating Resistance Field")
        print("=" * 50)
        
        # Step 1: Extract gene expression values
        print("\n1. Extracting gene expression...")
        collagen, lox, mmp = self._extract_gene_values(adata)
        
        # Step 2: Calculate raw density
        print("\n2. Computing raw density...")
        self.raw_density = self._calculate_raw_density(collagen, lox, mmp)
        
        # Step 3: Apply smoothing if requested
        if smooth:
            print("\n3. Applying KNN smoothing...")
            self.raw_density = self._smooth_field(adata, self.raw_density)
        
        # Step 4: Normalize to [0, 1] using sigmoid
        print("\n4. Normalizing to resistance probability...")
        self.resistance = self._normalize_resistance(self.raw_density)
        
        # Store results
        if store_in_adata:
            adata.obs['raw_density'] = self.raw_density
            adata.obs['resistance'] = self.resistance
            
            # Store individual gene contributions
            for gene, values in self.gene_contributions.items():
                adata.obs[f'expr_{gene}'] = values
            
            # Store resistance categories
            adata.obs['resistance_category'] = self._categorize_resistance(
                self.resistance
            )
        
        # Print summary
        self._print_summary()
        
        return self.resistance
    
    def _extract_gene_values(
        self, 
        adata: AnnData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract expression values for mechanotyping genes.
        
        Returns:
            Tuple of (collagen, lox, mmp) expression arrays.
        """
        def get_gene_expr(gene_list: list, adata: AnnData) -> np.ndarray:
            """Get sum of expression for a list of genes."""
            available = [g for g in gene_list if g in adata.var_names]
            if not available:
                print(f"   Warning: No genes found from {gene_list}")
                return np.zeros(adata.n_obs)
            
            # Get expression matrix
            expr = adata[:, available].X
            if sparse.issparse(expr):
                expr = expr.toarray()
            
            # Sum across genes (if multiple)
            total = np.sum(expr, axis=1).flatten()
            
            # Store individual contributions
            for g in available:
                g_expr = adata[:, g].X
                if sparse.issparse(g_expr):
                    g_expr = g_expr.toarray()
                self.gene_contributions[g] = g_expr.flatten()
            
            print(f"   {gene_list}: {len(available)} genes found, "
                  f"mean expr = {np.mean(total):.2f}")
            return total
        
        # Get collagen expression (COL1A1 + COL1A2)
        collagen = get_gene_expr(self.genes.collagen_genes, adata)
        
        # Get cross-linker expression (LOX + LOXL2)
        lox = get_gene_expr(self.genes.crosslinker_genes, adata)
        
        # Get MMP expression (MMP9 + MMP2)
        mmp = get_gene_expr(self.genes.degradation_genes, adata)
        
        return collagen, lox, mmp
    
    def _calculate_raw_density(
        self,
        collagen: np.ndarray,
        lox: np.ndarray,
        mmp: np.ndarray
    ) -> np.ndarray:
        """
        Calculate raw density score using the physics equation.
        
        D_i = (α * Collagen) × (1 + β * LOX) - (γ * MMP)
        
        The multiplication between Collagen and LOX is crucial:
        - High Collagen without LOX = weak (loose rope)
        - High Collagen with LOX = strong wall (cross-linked)
        """
        alpha = self.params.alpha
        beta = self.params.beta
        gamma = self.params.gamma
        
        # The physics equation
        # Material term: α * (COL1A1 + COL1A2)
        material = alpha * collagen
        
        # Cross-linking multiplier: (1 + β * LOX)
        # This amplifies the effect when LOX is high
        crosslink_factor = 1 + beta * lox
        
        # Combined structural term
        structural = material * crosslink_factor
        
        # Degradation term: γ * MMP
        degradation = gamma * mmp
        
        # Final density: structure minus degradation
        density = structural - degradation
        
        print(f"   Raw density range: [{density.min():.2f}, {density.max():.2f}]")
        print(f"   Mean: {density.mean():.2f}, Std: {density.std():.2f}")
        
        return density
    
    def _smooth_field(
        self,
        adata: AnnData,
        field: np.ndarray
    ) -> np.ndarray:
        """
        Apply KNN smoothing to handle zero-inflation.
        
        Uses the spatial neighbors to average out zeros.
        """
        from sklearn.neighbors import NearestNeighbors
        
        k = self.params.knn_smoothing
        
        # Get spatial coordinates
        coords = adata.obsm['spatial']
        
        # Build KNN graph
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Smooth by averaging with neighbors
        smoothed = np.zeros_like(field)
        for i in range(len(field)):
            neighbor_indices = indices[i]  # Includes self
            smoothed[i] = np.mean(field[neighbor_indices])
        
        print(f"   Smoothed with k={k} neighbors")
        print(f"   Smoothed range: [{smoothed.min():.2f}, {smoothed.max():.2f}]")
        
        return smoothed
    
    def _normalize_resistance(self, density: np.ndarray) -> np.ndarray:
        """
        Normalize raw density to resistance probability [0, 1].
        
        Uses sigmoid function: R = 1 / (1 + exp(-(D - μ)))
        """
        # Calculate or use provided center
        if self.params.sigmoid_center is None:
            mu = np.mean(density)
        else:
            mu = self.params.sigmoid_center
        
        # Apply sigmoid
        # expit is the scipy name for sigmoid: 1 / (1 + exp(-x))
        resistance = expit(density - mu)
        
        print(f"   Sigmoid center (μ): {mu:.2f}")
        print(f"   Resistance range: [{resistance.min():.3f}, {resistance.max():.3f}]")
        
        return resistance
    
    def _categorize_resistance(self, resistance: np.ndarray) -> np.ndarray:
        """
        Categorize resistance into biological interpretations.
        
        Returns array of category labels.
        """
        wall_thresh = self.params.wall_threshold
        fluid_thresh = self.params.fluid_threshold
        
        categories = np.empty(len(resistance), dtype=object)
        categories[:] = 'normal'
        categories[resistance > wall_thresh] = 'wall'
        categories[resistance < fluid_thresh] = 'fluid'
        
        # Count
        n_wall = (resistance > wall_thresh).sum()
        n_fluid = (resistance < fluid_thresh).sum()
        n_normal = len(resistance) - n_wall - n_fluid
        
        print(f"\n   Categories:")
        print(f"   - Wall (R > {wall_thresh}): {n_wall} spots ({100*n_wall/len(resistance):.1f}%)")
        print(f"   - Normal: {n_normal} spots ({100*n_normal/len(resistance):.1f}%)")
        print(f"   - Fluid (R < {fluid_thresh}): {n_fluid} spots ({100*n_fluid/len(resistance):.1f}%)")
        
        return categories
    
    def _print_summary(self) -> None:
        """Print final summary."""
        print("\n" + "=" * 50)
        print("Mechanotyping complete!")
        print("=" * 50)
        print(f"Resistance stored in adata.obs['resistance']")
        print(f"Categories stored in adata.obs['resistance_category']")
    
    def simulate_drug(
        self,
        adata: AnnData,
        target_gene: str,
        reduction_factor: float = 0.0
    ) -> np.ndarray:
        """
        Simulate the effect of a drug that targets a specific gene.
        
        This is the "Virtual Drug Test" for validation.
        
        Args:
            adata: AnnData object.
            target_gene: Gene to target (e.g., 'LOX' for LOX inhibitor).
            reduction_factor: Factor to reduce expression (0 = complete knockout).
            
        Returns:
            New resistance array after drug simulation.
        """
        print(f"\n{'='*50}")
        print(f"SIMULATING DRUG: {target_gene} inhibitor")
        print(f"Reduction factor: {reduction_factor}")
        print(f"{'='*50}")
        
        # Make a copy of the data
        adata_sim = adata.copy()
        
        # Reduce target gene expression
        if target_gene in adata_sim.var_names:
            original_expr = adata_sim[:, target_gene].X.copy()
            if sparse.issparse(original_expr):
                original_expr = original_expr.toarray()
            
            # Apply reduction
            new_expr = original_expr * reduction_factor
            
            # Update in adata (this is a simulation, not modifying original)
            print(f"Original {target_gene} mean: {original_expr.mean():.2f}")
            print(f"Simulated {target_gene} mean: {new_expr.mean():.2f}")
        else:
            print(f"Warning: {target_gene} not found in data")
            return self.resistance
        
        # Recalculate resistance
        collagen, lox, mmp = self._extract_gene_values(adata_sim)
        
        # If targeting LOX, use the reduced values
        if target_gene in self.genes.crosslinker_genes:
            lox = lox * reduction_factor
        elif target_gene in self.genes.degradation_genes:
            mmp = mmp * reduction_factor
        
        sim_density = self._calculate_raw_density(collagen, lox, mmp)
        sim_resistance = self._normalize_resistance(sim_density)
        
        # Compare
        print(f"\nOriginal mean resistance: {self.resistance.mean():.3f}")
        print(f"Simulated mean resistance: {sim_resistance.mean():.3f}")
        print(f"Change: {(sim_resistance.mean() - self.resistance.mean())*100:.1f}%")
        
        return sim_resistance


def calculate_resistance_field(
    adata: AnnData,
    config: Optional[Config] = None,
    **kwargs
) -> np.ndarray:
    """
    Convenience function for resistance calculation.
    
    Args:
        adata: AnnData object with gene expression.
        config: Configuration object.
        **kwargs: Additional arguments passed to Mechanotyper.calculate_resistance().
        
    Returns:
        Resistance array.
    """
    mechanotyper = Mechanotyper(config)
    return mechanotyper.calculate_resistance(adata, **kwargs)
