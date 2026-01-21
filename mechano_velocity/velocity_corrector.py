"""
Physics-constrained velocity correction module.

Combines RNA velocity (biological intent) with ECM resistance (physical constraints)
to produce corrected velocity vectors that respect tissue barriers.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Tuple, Dict, Any
from anndata import AnnData

from .config import Config, default_config
from .graph_builder import GraphBuilder


class VelocityCorrector:
    """
    Apply physics-based corrections to RNA velocity vectors.
    
    The correction logic:
    1. Get naive velocity from scVelo (where cells WANT to go)
    2. Apply resistance penalty (where cells CAN go)
    3. Project velocity onto allowed directions
    
    The final corrected velocity:
    v_corrected[i] = Σ W_ij × (x_j - x_i) for j in neighbors
    
    Where W_ij = Similarity(i,j) × (1 - R_j)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize velocity corrector.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or default_config
        self.naive_velocity: Optional[np.ndarray] = None
        self.corrected_velocity: Optional[np.ndarray] = None
        self.velocity_magnitude: Optional[np.ndarray] = None
        
    def compute_rna_velocity(
        self,
        adata: AnnData,
        mode: str = 'stochastic',
        **scvelo_kwargs
    ) -> np.ndarray:
        """
        Compute RNA velocity using scVelo.
        
        Args:
            adata: AnnData object with spliced/unspliced counts.
            mode: Velocity mode ('stochastic', 'deterministic', 'dynamical').
            **scvelo_kwargs: Additional arguments for scVelo.
            
        Returns:
            Velocity matrix (n_spots × n_genes).
        """
        try:
            import scvelo as scv
        except ImportError:
            raise ImportError(
                "scVelo is required for RNA velocity. "
                "Install with: pip install scvelo"
            )
        
        print("=" * 50)
        print("COMPUTING RNA VELOCITY")
        print("=" * 50)
        
        # Check if velocity already computed
        if 'velocity' in adata.layers:
            print("Using pre-computed velocity from adata.layers['velocity']")
            return adata.layers['velocity']
        
        # Standard scVelo pipeline
        print("\n1. Computing moments...")
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        
        print(f"\n2. Computing {mode} velocity...")
        if mode == 'dynamical':
            scv.tl.recover_dynamics(adata, **scvelo_kwargs)
            scv.tl.velocity(adata, mode='dynamical')
        else:
            scv.tl.velocity(adata, mode=mode, **scvelo_kwargs)
        
        print("\n3. Computing velocity graph...")
        scv.tl.velocity_graph(adata)
        
        self.naive_velocity = adata.layers['velocity']
        
        print("\nRNA Velocity computed successfully!")
        
        return self.naive_velocity
    
    def apply_resistance_correction(
        self,
        adata: AnnData,
        graph_builder: Optional[GraphBuilder] = None,
        method: str = 'projection'
    ) -> np.ndarray:
        """
        Apply physics-based resistance correction to velocity.
        
        Args:
            adata: AnnData object with velocity and resistance.
            graph_builder: Pre-built GraphBuilder, or None to build new.
            method: Correction method ('projection', 'scaling', 'hard_threshold').
            
        Returns:
            Corrected velocity embedding (n_spots × 2).
        """
        print("=" * 50)
        print("APPLYING RESISTANCE CORRECTION")
        print("=" * 50)
        
        # Validate inputs
        if 'resistance' not in adata.obs.columns:
            raise ValueError("Resistance not computed. Run Mechanotyper first.")
        
        # Build or get graph
        if graph_builder is None:
            graph_builder = GraphBuilder(self.config)
            graph_builder.build_spatial_graph(
                adata, 
                include_resistance=True,
                include_similarity=True
            )
        
        adjacency = graph_builder.adjacency
        if adjacency is None:
            adjacency = adata.obsp.get('spatial_connectivities')
        
        if adjacency is None:
            raise ValueError("No spatial graph found. Build graph first.")
        
        # Get spatial coordinates
        coords = adata.obsm['spatial']
        n_spots = len(coords)
        
        print(f"\n1. Computing corrected velocity vectors ({method})...")
        
        if method == 'projection':
            corrected = self._projection_correction(coords, adjacency)
        elif method == 'scaling':
            corrected = self._scaling_correction(adata, coords, adjacency)
        elif method == 'hard_threshold':
            corrected = self._threshold_correction(adata, coords, adjacency)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.corrected_velocity = corrected
        
        # Compute magnitude
        self.velocity_magnitude = np.linalg.norm(corrected, axis=1)
        
        # Store in adata
        adata.obsm['velocity_corrected'] = corrected
        adata.obs['velocity_magnitude'] = self.velocity_magnitude
        
        # Compute and store velocity direction (angle)
        angles = np.arctan2(corrected[:, 1], corrected[:, 0])
        adata.obs['velocity_angle'] = angles
        
        self._print_summary()
        
        return corrected
    
    def _projection_correction(
        self,
        coords: np.ndarray,
        adjacency: sparse.csr_matrix
    ) -> np.ndarray:
        """
        Projection-based correction.
        
        v_corrected[i] = Σ W_ij × (x_j - x_i) / Σ W_ij
        
        The velocity vector is the weighted average of allowed directions.
        """
        n_spots = len(coords)
        corrected = np.zeros((n_spots, 2))
        
        for i in range(n_spots):
            # Get neighbors and weights
            row = adjacency.getrow(i)
            neighbors = row.indices
            weights = row.data
            
            if len(neighbors) == 0:
                continue
            
            # Compute direction vectors to neighbors
            directions = coords[neighbors] - coords[i]
            
            # Weighted average
            weight_sum = weights.sum()
            if weight_sum > 0:
                corrected[i] = np.average(directions, weights=weights, axis=0)
        
        return corrected
    
    def _scaling_correction(
        self,
        adata: AnnData,
        coords: np.ndarray,
        adjacency: sparse.csr_matrix
    ) -> np.ndarray:
        """
        Scaling-based correction.
        
        Scale the naive velocity magnitude by (1 - local_resistance).
        Direction preserved, magnitude reduced in high-resistance areas.
        """
        # Start with projection-based direction
        base_velocity = self._projection_correction(coords, adjacency)
        
        # Get local resistance
        resistance = adata.obs['resistance'].values
        
        # Scale magnitude by (1 - R)
        scaling = 1 - resistance
        scaling = np.clip(scaling, 0.01, 1.0)  # Avoid zero
        
        # Apply scaling
        magnitude = np.linalg.norm(base_velocity, axis=1, keepdims=True)
        magnitude = np.maximum(magnitude, 1e-6)  # Avoid division by zero
        direction = base_velocity / magnitude
        
        scaled = direction * magnitude * scaling.reshape(-1, 1)
        
        return scaled
    
    def _threshold_correction(
        self,
        adata: AnnData,
        coords: np.ndarray,
        adjacency: sparse.csr_matrix
    ) -> np.ndarray:
        """
        Hard threshold correction.
        
        Zero out velocity for spots with resistance above threshold.
        """
        base_velocity = self._projection_correction(coords, adjacency)
        
        # Get resistance
        resistance = adata.obs['resistance'].values
        threshold = self.config.mechanotyping.wall_threshold
        
        # Zero out high-resistance spots
        mask = resistance > threshold
        base_velocity[mask] = 0
        
        print(f"   Zeroed {mask.sum()} spots above threshold {threshold}")
        
        return base_velocity
    
    def _print_summary(self) -> None:
        """Print velocity correction summary."""
        print("\n" + "=" * 50)
        print("Velocity Correction Complete!")
        print("=" * 50)
        if self.velocity_magnitude is not None:
            print(f"Mean velocity magnitude: {self.velocity_magnitude.mean():.4f}")
            print(f"Max velocity magnitude: {self.velocity_magnitude.max():.4f}")
            print(f"Stalled spots (mag < 0.01): {(self.velocity_magnitude < 0.01).sum()}")
    
    def compare_velocities(
        self,
        adata: AnnData
    ) -> Dict[str, Any]:
        """
        Compare naive vs corrected velocities.
        
        Args:
            adata: AnnData with both velocity types.
            
        Returns:
            Dictionary with comparison metrics.
        """
        if 'velocity_corrected' not in adata.obsm:
            raise ValueError("Corrected velocity not computed yet.")
        
        corrected = adata.obsm['velocity_corrected']
        corrected_mag = np.linalg.norm(corrected, axis=1)
        
        # Get resistance for correlation
        resistance = adata.obs['resistance'].values
        
        # Compute correlation
        correlation = np.corrcoef(resistance, corrected_mag)[0, 1]
        
        # Count affected spots
        high_res_spots = resistance > self.config.mechanotyping.wall_threshold
        mean_mag_high_res = corrected_mag[high_res_spots].mean() if high_res_spots.sum() > 0 else 0
        mean_mag_low_res = corrected_mag[~high_res_spots].mean() if (~high_res_spots).sum() > 0 else 0
        
        result = {
            'resistance_velocity_correlation': correlation,
            'mean_velocity_high_resistance': mean_mag_high_res,
            'mean_velocity_low_resistance': mean_mag_low_res,
            'velocity_reduction_ratio': mean_mag_high_res / max(mean_mag_low_res, 1e-6),
            'n_high_resistance_spots': int(high_res_spots.sum()),
            'n_low_resistance_spots': int((~high_res_spots).sum()),
        }
        
        print("\nVelocity Comparison:")
        print(f"  Resistance-Velocity correlation: {correlation:.3f}")
        print(f"  Mean velocity (high R): {mean_mag_high_res:.4f}")
        print(f"  Mean velocity (low R): {mean_mag_low_res:.4f}")
        print(f"  Reduction ratio: {result['velocity_reduction_ratio']:.3f}")
        
        return result
    
    def identify_trapped_cells(
        self,
        adata: AnnData,
        velocity_threshold: float = 0.01,
        resistance_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Identify cells that are trapped (high resistance, low velocity).
        
        These are cells that WANT to move (express migration genes)
        but CANNOT move (surrounded by ECM).
        
        Args:
            adata: AnnData with velocity and resistance.
            velocity_threshold: Below this magnitude = stalled.
            resistance_threshold: Above this = blocked. Uses config default if None.
            
        Returns:
            Boolean mask of trapped cells.
        """
        if resistance_threshold is None:
            resistance_threshold = self.config.mechanotyping.wall_threshold
        
        # Get values
        velocity_mag = adata.obs.get('velocity_magnitude')
        if velocity_mag is None:
            if 'velocity_corrected' in adata.obsm:
                velocity_mag = np.linalg.norm(adata.obsm['velocity_corrected'], axis=1)
            else:
                raise ValueError("Velocity not computed yet.")
        
        resistance = adata.obs['resistance'].values
        
        # Trapped = high resistance AND low velocity
        trapped = (resistance > resistance_threshold) & (velocity_mag < velocity_threshold)
        
        # Store result
        adata.obs['is_trapped'] = trapped
        
        print(f"\nTrapped cell analysis:")
        print(f"  Total trapped: {trapped.sum()} ({100*trapped.mean():.1f}%)")
        print(f"  Criteria: R > {resistance_threshold}, |v| < {velocity_threshold}")
        
        return trapped


def correct_velocity(
    adata: AnnData,
    config: Optional[Config] = None,
    **kwargs
) -> np.ndarray:
    """
    Convenience function for velocity correction.
    
    Args:
        adata: AnnData object.
        config: Configuration object.
        **kwargs: Additional arguments.
        
    Returns:
        Corrected velocity embedding.
    """
    corrector = VelocityCorrector(config)
    return corrector.apply_resistance_correction(adata, **kwargs)
