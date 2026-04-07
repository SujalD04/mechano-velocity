"""
Mechano-Velocity: Physics-Informed Graph Neural Network for Spatial Transcriptomics

A computational framework that corrects "False Positive" migration predictions
by detecting physical barriers (ECM/collagen) and applying resistance penalties
to RNA velocity vectors.
"""

__version__ = "0.2.0"
__author__ = "Mechano-Velocity Team"

from .config import Config
from .data_loader import DataLoader
from .preprocessor import Preprocessor
from .mechanotyper import Mechanotyper
from .graph_builder import GraphBuilder
from .velocity_corrector import VelocityCorrector
from .clinical_scorer import ClinicalScorer
from .visualizer import Visualizer
from .database import DatabaseManager

# GNN components (optional — requires torch-geometric)
try:
    from .gnn_model import (
        MechanoVelocityGNN,
        PhysicsLoss,
        prepare_gnn_data,
        load_trained_model,
        predict_velocity,
    )
    HAS_GNN = True
except ImportError:
    HAS_GNN = False

__all__ = [
    "Config",
    "DataLoader", 
    "Preprocessor",
    "Mechanotyper",
    "GraphBuilder",
    "VelocityCorrector",
    "ClinicalScorer",
    "Visualizer",
    "DatabaseManager",
]

