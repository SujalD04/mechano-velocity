"""
Configuration and hyperparameters for Mechano-Velocity.

Contains all biological constants, algorithm parameters, and file paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import json


@dataclass
class GeneSignature:
    """Gene signature weights for mechanotyping."""
    # Construction Team (Barrier Builders)
    collagen_genes: List[str] = field(default_factory=lambda: ["COL1A1", "COL1A2"])
    crosslinker_genes: List[str] = field(default_factory=lambda: ["LOX", "LOXL2"])
    scaffold_genes: List[str] = field(default_factory=lambda: ["FN1"])
    
    # Demolition Team (Tunnel Borers)  
    degradation_genes: List[str] = field(default_factory=lambda: ["MMP9", "MMP2"])
    
    # Travelers (Mobile Agents)
    tcell_markers: List[str] = field(default_factory=lambda: ["CD8A", "CD8B", "CD3E"])
    tumor_markers: List[str] = field(default_factory=lambda: ["EPCAM", "KRT19", "KRT8"])
    
    @property
    def all_genes(self) -> List[str]:
        """Return all genes needed for analysis."""
        return (
            self.collagen_genes + 
            self.crosslinker_genes + 
            self.scaffold_genes +
            self.degradation_genes + 
            self.tcell_markers + 
            self.tumor_markers
        )


@dataclass
class MechanotypingParams:
    """Parameters for the resistance calculation equation."""
    # D_i = (α * COL1A1 + α * COL1A2) × (1 + β * LOX) - (γ * MMP9)
    alpha: float = 1.0      # Collagen weight
    beta: float = 0.5       # LOX cross-linking multiplier
    gamma: float = 0.8      # MMP degradation weight
    
    # Sigmoid normalization centering
    sigmoid_center: Optional[float] = None  # Auto-calculated from data mean
    
    # Smoothing for zero-inflation
    knn_smoothing: int = 6  # Number of neighbors for smoothing
    
    # Resistance thresholds
    wall_threshold: float = 0.8     # R > 0.8 = Impassable
    fluid_threshold: float = 0.2   # R < 0.2 = Open space


@dataclass  
class CellPhysics:
    """Physical constraints for different cell types."""
    # Nucleus sizes (determines minimum pore size for passage)
    tcell_nucleus_um: float = 5.0       # T-cells can squeeze through 5µm gaps
    tumor_nucleus_um: float = 12.5      # Tumor cells need 10-15µm gaps (average)
    
    # Visium resolution
    spot_diameter_um: float = 55.0      # One spot = 55µm diameter
    spot_spacing_um: float = 100.0      # Center-to-center = 100µm
    
    # Hexagonal grid connectivity
    neighbors_per_spot: int = 6


@dataclass
class PreprocessingParams:
    """Parameters for data preprocessing."""
    min_counts: int = 500           # Minimum UMI counts per spot
    min_genes: int = 200            # Minimum genes per spot
    target_sum: float = 1e4         # CPM normalization target
    n_top_genes: int = 2000         # For highly variable genes
    n_pcs: int = 50                 # Principal components
    n_neighbors: int = 15           # For neighborhood graph


@dataclass
class VisualizationParams:
    """Visualization settings."""
    figsize: tuple = (12, 10)
    dpi: int = 150
    cmap_resistance: str = "RdBu_r"     # Red = wall, Blue = fluid
    cmap_velocity: str = "coolwarm"
    streamplot_density: float = 1.5
    alpha_overlay: float = 0.6


@dataclass
class DatabaseConfig:
    """Database configuration for storing outputs."""
    db_path: str = "mechano_velocity.db"
    table_prefix: str = "mv_"


@dataclass
class Config:
    """Master configuration class."""
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "output")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    
    # Dataset name
    dataset_name: str = "V1_Breast_Cancer_Block_A"
    
    # Sub-configurations
    genes: GeneSignature = field(default_factory=GeneSignature)
    mechanotyping: MechanotypingParams = field(default_factory=MechanotypingParams)
    cell_physics: CellPhysics = field(default_factory=CellPhysics)
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)
    visualization: VisualizationParams = field(default_factory=VisualizationParams)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    @property
    def dataset_path(self) -> Path:
        """Full path to the dataset folder."""
        return self.data_dir / self.dataset_name
    
    @property
    def h5_path(self) -> Path:
        """Path to the filtered feature matrix."""
        return self.dataset_path / "filtered_feature_bc_matrix.h5"
    
    @property
    def spatial_path(self) -> Path:
        """Path to the spatial folder."""
        return self.dataset_path / "spatial"
    
    def validate_dataset(self) -> bool:
        """Check if all required dataset files exist."""
        required_files = [
            self.h5_path,
            self.spatial_path / "tissue_hires_image.png",
            self.spatial_path / "scalefactors_json.json",
        ]
        missing = [f for f in required_files if not f.exists()]
        if missing:
            print(f"Missing files: {missing}")
            return False
        return True
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            "version": "0.1.0",
            "dataset_name": self.dataset_name,
            "mechanotyping": {
                "alpha": self.mechanotyping.alpha,
                "beta": self.mechanotyping.beta,
                "gamma": self.mechanotyping.gamma,
            },
            "cell_physics": {
                "tcell_nucleus_um": self.cell_physics.tcell_nucleus_um,
                "tumor_nucleus_um": self.cell_physics.tumor_nucleus_um,
            },
            "preprocessing": {
                "min_counts": self.preprocessing.min_counts,
                "target_sum": self.preprocessing.target_sum,
            }
        }
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON."""
        if path is None:
            path = self.output_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls()
        config.dataset_name = data.get("dataset_name", config.dataset_name)
        if "mechanotyping" in data:
            config.mechanotyping.alpha = data["mechanotyping"].get("alpha", 1.0)
            config.mechanotyping.beta = data["mechanotyping"].get("beta", 0.5)
            config.mechanotyping.gamma = data["mechanotyping"].get("gamma", 0.8)
        return config


# Default configuration instance
default_config = Config()
