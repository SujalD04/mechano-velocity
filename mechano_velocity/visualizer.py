"""
Visualization utilities for Mechano-Velocity.

Provides publication-quality plots for:
- Resistance heatmaps
- Velocity streamplots
- Before/after comparisons
- Histological overlays
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Tuple, List, Union
from pathlib import Path
from anndata import AnnData

from .config import Config, default_config


class Visualizer:
    """
    Publication-quality visualization for spatial transcriptomics.
    
    All plots support:
    - Spatial context (tissue coordinates)
    - Color mapping with legends
    - Overlay with H&E images
    - Export to various formats
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or default_config
        self.viz_params = self.config.visualization
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_resistance_heatmap(
        self,
        adata: AnnData,
        ax: Optional[plt.Axes] = None,
        show_image: bool = True,
        spot_size: float = 50,
        alpha: float = 0.8,
        title: str = "ECM Resistance Field",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the resistance field as a spatial heatmap.
        
        Colors:
        - Red: High resistance (Wall)
        - Blue: Low resistance (Fluid)
        
        Args:
            adata: AnnData with resistance computed.
            ax: Matplotlib axes. Creates new figure if None.
            show_image: Whether to overlay on H&E image.
            spot_size: Size of spots in the plot.
            alpha: Transparency of spots.
            title: Plot title.
            save_path: Path to save figure.
            
        Returns:
            Matplotlib Figure object.
        """
        if 'resistance' not in adata.obs.columns:
            raise ValueError("Resistance not computed. Run Mechanotyper first.")
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.viz_params.figsize)
        else:
            fig = ax.figure
        
        # Get coordinates and values
        coords = adata.obsm['spatial']
        resistance = adata.obs['resistance'].values
        
        # Plot H&E background if available and requested
        if show_image and 'spatial' in adata.uns and 'images' in adata.uns['spatial']:
            self._plot_tissue_image(adata, ax)
        
        # Create scatter plot
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=resistance,
            cmap=self.viz_params.cmap_resistance,
            s=spot_size,
            alpha=alpha,
            vmin=0,
            vmax=1,
            edgecolors='none'
        )
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('Resistance (0=Fluid, 1=Wall)', fontsize=10)
        
        # Labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Spatial X (µm)')
        ax.set_ylabel('Spatial Y (µm)')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match image coordinates
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.viz_params.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_velocity_streamplot(
        self,
        adata: AnnData,
        velocity_key: str = 'velocity_corrected',
        ax: Optional[plt.Axes] = None,
        color_by: str = 'resistance',
        arrow_scale: float = 1.0,
        grid_resolution: int = 30,
        title: str = "Physics-Constrained Velocity Field",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot velocity as a streamplot.
        
        Args:
            adata: AnnData with velocity computed.
            velocity_key: Key in adata.obsm for velocity.
            ax: Matplotlib axes.
            color_by: Color streamlines by this variable.
            arrow_scale: Scale factor for arrows.
            grid_resolution: Resolution of interpolation grid.
            title: Plot title.
            save_path: Path to save figure.
            
        Returns:
            Matplotlib Figure object.
        """
        if velocity_key not in adata.obsm:
            raise ValueError(f"Velocity '{velocity_key}' not found in adata.obsm")
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.viz_params.figsize)
        else:
            fig = ax.figure
        
        # Get data
        coords = adata.obsm['spatial']
        velocity = adata.obsm[velocity_key]
        
        # Create regular grid for streamplot
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Grid bounds
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        xi = np.linspace(x_min, x_max, grid_resolution)
        yi = np.linspace(y_min, y_max, grid_resolution)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate velocity onto grid
        from scipy.interpolate import griddata
        U = griddata((x, y), velocity[:, 0], (Xi, Yi), method='linear', fill_value=0)
        V = griddata((x, y), velocity[:, 1], (Xi, Yi), method='linear', fill_value=0)
        
        # Color by variable
        if color_by == 'resistance' and 'resistance' in adata.obs.columns:
            color_values = adata.obs['resistance'].values
            C = griddata((x, y), color_values, (Xi, Yi), method='linear', fill_value=0.5)
            cmap = self.viz_params.cmap_resistance
        elif color_by == 'magnitude':
            C = np.sqrt(U**2 + V**2)
            cmap = 'viridis'
        else:
            C = np.ones_like(U)
            cmap = 'gray'
        
        # Speed for line width
        speed = np.sqrt(U**2 + V**2)
        lw = 2 * speed / (speed.max() + 1e-6) + 0.5
        
        # Plot streamlines
        strm = ax.streamplot(
            xi, yi, U, V,
            color=C,
            cmap=cmap,
            linewidth=lw,
            density=self.viz_params.streamplot_density,
            arrowsize=arrow_scale,
        )
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(strm.lines, cax=cax)
        cbar.set_label(color_by.title(), fontsize=10)
        
        # Labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Spatial X (µm)')
        ax.set_ylabel('Spatial Y (µm)')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.viz_params.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_velocity_arrows(
        self,
        adata: AnnData,
        velocity_key: str = 'velocity_corrected',
        ax: Optional[plt.Axes] = None,
        color_by: str = 'resistance',
        arrow_scale: float = 0.3,
        subsample: int = 1,
        title: str = "Velocity Vectors",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot velocity as quiver (arrow) plot.
        
        Args:
            adata: AnnData with velocity.
            velocity_key: Key in adata.obsm.
            ax: Matplotlib axes.
            color_by: Color arrows by this variable.
            arrow_scale: Scale factor for arrows.
            subsample: Plot every Nth arrow.
            title: Plot title.
            save_path: Save path.
            
        Returns:
            Figure object.
        """
        if velocity_key not in adata.obsm:
            raise ValueError(f"Velocity '{velocity_key}' not found.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.viz_params.figsize)
        else:
            fig = ax.figure
        
        # Get data (subsample)
        idx = np.arange(0, adata.n_obs, subsample)
        coords = adata.obsm['spatial'][idx]
        velocity = adata.obsm[velocity_key][idx]
        
        # Color
        if color_by == 'resistance' and 'resistance' in adata.obs.columns:
            colors = adata.obs['resistance'].values[idx]
            cmap = self.viz_params.cmap_resistance
        elif color_by == 'magnitude':
            colors = np.linalg.norm(velocity, axis=1)
            cmap = 'viridis'
        else:
            colors = None
            cmap = None
        
        # Plot quiver
        quiver = ax.quiver(
            coords[:, 0],
            coords[:, 1],
            velocity[:, 0],
            velocity[:, 1],
            colors,
            cmap=cmap,
            scale=1/arrow_scale,
            alpha=0.8,
            width=0.003
        )
        
        if colors is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(quiver, cax=cax)
            cbar.set_label(color_by.title(), fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Spatial X (µm)')
        ax.set_ylabel('Spatial Y (µm)')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.viz_params.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_comparison(
        self,
        adata: AnnData,
        naive_key: str = 'X_pca',
        corrected_key: str = 'velocity_corrected',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Side-by-side comparison of naive vs corrected velocity.
        
        Args:
            adata: AnnData with both velocities.
            naive_key: Key for naive velocity/embedding.
            corrected_key: Key for corrected velocity.
            save_path: Save path.
            
        Returns:
            Figure object.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Resistance map
        self.plot_resistance_heatmap(
            adata, 
            ax=axes[0],
            show_image=False,
            title="ECM Resistance Field"
        )
        
        # Plot 2: Corrected velocity
        if corrected_key in adata.obsm:
            self.plot_velocity_arrows(
                adata,
                velocity_key=corrected_key,
                ax=axes[1],
                color_by='resistance',
                title="Corrected Velocity (Physics-Constrained)"
            )
        else:
            axes[1].text(0.5, 0.5, "Corrected velocity not found",
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title("Corrected Velocity")
        
        # Plot 3: Cell type overlay
        self._plot_cell_types(adata, axes[2])
        
        plt.suptitle("Mechano-Velocity Analysis Overview", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.viz_params.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def _plot_tissue_image(self, adata: AnnData, ax: plt.Axes) -> None:
        """Plot the H&E tissue image as background."""
        try:
            # Get image from adata
            library_id = list(adata.uns['spatial'].keys())[0]
            img = adata.uns['spatial'][library_id]['images']['hires']
            scale = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
            
            # Get extent
            coords = adata.obsm['spatial']
            extent = [
                coords[:, 0].min() - 100,
                coords[:, 0].max() + 100,
                coords[:, 1].max() + 100,
                coords[:, 1].min() - 100
            ]
            
            ax.imshow(img, extent=extent, alpha=0.3, aspect='auto')
        except (KeyError, IndexError):
            pass  # No image available
    
    def _plot_cell_types(self, adata: AnnData, ax: plt.Axes) -> None:
        """Plot cell type annotations."""
        coords = adata.obsm['spatial']
        
        # Default colors
        colors = np.full(adata.n_obs, 'lightgray', dtype=object)
        
        # Color by cell type if available
        if 'is_tumor' in adata.obs.columns:
            colors[adata.obs['is_tumor'].values] = 'red'
        if 'is_tcell' in adata.obs.columns:
            colors[adata.obs['is_tcell'].values] = 'blue'
        if 'is_boundary' in adata.obs.columns:
            colors[adata.obs['is_boundary'].values] = 'orange'
        if 'is_trapped' in adata.obs.columns:
            colors[adata.obs['is_trapped'].values] = 'purple'
        
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=30, alpha=0.7)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Tumor'),
            Patch(facecolor='blue', label='T-cells'),
            Patch(facecolor='orange', label='Boundary'),
            Patch(facecolor='purple', label='Trapped'),
            Patch(facecolor='lightgray', label='Other'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        ax.set_title("Cell Type Annotations", fontsize=14, fontweight='bold')
        ax.set_xlabel('Spatial X (µm)')
        ax.set_ylabel('Spatial Y (µm)')
        ax.set_aspect('equal')
        ax.invert_yaxis()
    
    def plot_drug_simulation(
        self,
        adata: AnnData,
        original_resistance: np.ndarray,
        simulated_resistance: np.ndarray,
        drug_name: str = "LOX Inhibitor",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot before/after drug simulation.
        
        Args:
            adata: AnnData object.
            original_resistance: Original resistance values.
            simulated_resistance: Post-drug resistance values.
            drug_name: Name of simulated drug.
            save_path: Save path.
            
        Returns:
            Figure object.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        coords = adata.obsm['spatial']
        
        # Before
        sc1 = axes[0].scatter(
            coords[:, 0], coords[:, 1],
            c=original_resistance,
            cmap=self.viz_params.cmap_resistance,
            s=30, alpha=0.8, vmin=0, vmax=1
        )
        axes[0].set_title("Before Treatment", fontsize=12, fontweight='bold')
        plt.colorbar(sc1, ax=axes[0], label='Resistance')
        
        # After
        sc2 = axes[1].scatter(
            coords[:, 0], coords[:, 1],
            c=simulated_resistance,
            cmap=self.viz_params.cmap_resistance,
            s=30, alpha=0.8, vmin=0, vmax=1
        )
        axes[1].set_title(f"After {drug_name}", fontsize=12, fontweight='bold')
        plt.colorbar(sc2, ax=axes[1], label='Resistance')
        
        # Difference
        diff = simulated_resistance - original_resistance
        sc3 = axes[2].scatter(
            coords[:, 0], coords[:, 1],
            c=diff,
            cmap='RdBu',
            s=30, alpha=0.8,
            vmin=-0.5, vmax=0.5
        )
        axes[2].set_title("Change in Resistance", fontsize=12, fontweight='bold')
        plt.colorbar(sc3, ax=axes[2], label='ΔResistance')
        
        for ax in axes:
            ax.set_xlabel('Spatial X (µm)')
            ax.set_ylabel('Spatial Y (µm)')
            ax.set_aspect('equal')
            ax.invert_yaxis()
        
        plt.suptitle(f"Virtual Drug Test: {drug_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.viz_params.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig


def plot_analysis(
    adata: AnnData,
    output_dir: Optional[str] = None,
    config: Optional[Config] = None
) -> None:
    """
    Generate all standard plots.
    
    Args:
        adata: AnnData with analysis complete.
        output_dir: Directory to save plots.
        config: Configuration object.
    """
    viz = Visualizer(config)
    
    if output_dir is None:
        output_dir = config.output_dir if config else Path("output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    viz.plot_resistance_heatmap(adata, save_path=output_dir / "resistance_map.png")
    
    if 'velocity_corrected' in adata.obsm:
        viz.plot_velocity_streamplot(adata, save_path=output_dir / "velocity_streamplot.png")
        viz.plot_velocity_arrows(adata, save_path=output_dir / "velocity_arrows.png")
    
    viz.plot_comparison(adata, save_path=output_dir / "analysis_overview.png")
    
    print(f"\nAll plots saved to: {output_dir}")
