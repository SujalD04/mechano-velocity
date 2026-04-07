"""
Physics-Informed Graph Neural Network for velocity correction.

This GNN learns to predict corrected velocity vectors from node features
(gene expression, resistance) and graph structure (spatial adjacency),
replacing the hand-crafted weighted-average approach with a learned model.

Architecture:
    Input: Node features (PCA embeddings + resistance + spatial coords)
    → 3× GCN message-passing layers with residual connections
    → MLP head → 2D corrected velocity vector per node

Training signal:
    The model is supervised by the graph-diffusion velocity (the existing
    hand-crafted correction) and penalized for predicting high velocity
    in high-resistance regions (physics loss).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GraphNorm
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


# ================================================================
# GNN Architecture
# ================================================================
if HAS_TORCH_GEOMETRIC:

    class MechanoVelocityGNN(nn.Module):
        """
        Physics-Informed GNN for velocity correction.

        3-layer GCN with residual connections, batch normalization,
        and a physics-constrained loss function.
        """

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 128,
            out_channels: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()

            # Input projection
            self.input_proj = nn.Linear(in_channels, hidden_channels)

            # GCN layers with normalization
            self.conv1 = GCNConv(hidden_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)

            self.norm1 = GraphNorm(hidden_channels)
            self.norm2 = GraphNorm(hidden_channels)
            self.norm3 = GraphNorm(hidden_channels)

            # Output MLP
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, out_channels),
            )

            self.dropout = dropout

        def forward(self, data: Data) -> torch.Tensor:
            x, edge_index = data.x, data.edge_index
            edge_weight = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None

            # Input projection
            x = self.input_proj(x)
            x = F.relu(x)

            # GCN block 1
            residual = x
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = self.norm1(x)
            x = F.relu(x) + residual
            x = F.dropout(x, p=self.dropout, training=self.training)

            # GCN block 2
            residual = x
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            x = self.norm2(x)
            x = F.relu(x) + residual
            x = F.dropout(x, p=self.dropout, training=self.training)

            # GCN block 3
            residual = x
            x = self.conv3(x, edge_index, edge_weight=edge_weight)
            x = self.norm3(x)
            x = F.relu(x) + residual

            # Output
            velocity = self.mlp(x)
            return velocity


    class PhysicsLoss(nn.Module):
        """
        Combined loss for physics-informed training.

        L = L_velocity + λ_physics * L_physics

        L_velocity: MSE between predicted and target velocity
        L_physics:  penalty for predicting high velocity in high-resistance spots
                    = mean( R_i * ||v_i||^2 )
        """

        def __init__(self, lambda_physics: float = 0.5):
            super().__init__()
            self.lambda_physics = lambda_physics
            self.mse = nn.MSELoss()

        def forward(
            self,
            pred_velocity: torch.Tensor,
            target_velocity: torch.Tensor,
            resistance: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:

            # Velocity reconstruction loss
            loss_vel = self.mse(pred_velocity, target_velocity)

            # Physics constraint: high R should mean low velocity
            pred_mag_sq = (pred_velocity ** 2).sum(dim=1)
            loss_physics = (resistance * pred_mag_sq).mean()

            # Total
            total = loss_vel + self.lambda_physics * loss_physics

            metrics = {
                "loss_total": total.item(),
                "loss_velocity": loss_vel.item(),
                "loss_physics": loss_physics.item(),
            }

            return total, metrics


# ================================================================
# Training & Inference Utilities (used in both Colab and locally)
# ================================================================

def prepare_gnn_data(adata, n_pcs: int = 50) -> Dict[str, np.ndarray]:
    """
    Prepare node features, edge index, and targets from AnnData.

    Returns a dict of numpy arrays that can be saved/loaded as .npz.
    """
    from scipy import sparse as sp

    # --- Node features: PCA + resistance + spatial coords ---
    pca = adata.obsm["X_pca"][:, :n_pcs]  # (N, 50)
    resistance = adata.obs["resistance"].values.reshape(-1, 1)  # (N, 1)
    coords = adata.obsm["spatial"]  # (N, 2)

    # Normalize spatial coords to [0, 1]
    coords_norm = coords.copy().astype(np.float32)
    coords_norm -= coords_norm.min(axis=0)
    coords_norm /= coords_norm.max(axis=0) + 1e-8

    features = np.hstack([pca, resistance, coords_norm]).astype(np.float32)

    # --- Edge index from spatial graph ---
    adj = adata.obsp["spatial_connectivities"]
    rows, cols = adj.nonzero()
    edge_index = np.vstack([rows, cols]).astype(np.int64)
    edge_weight = np.array(adj[rows, cols]).flatten().astype(np.float32)

    # --- Target velocity ---
    target_velocity = adata.obsm["velocity_corrected"].astype(np.float32)

    return {
        "features": features,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "resistance": resistance.flatten().astype(np.float32),
        "target_velocity": target_velocity,
        "spatial_coords": coords.astype(np.float32),
    }


def save_gnn_data(data_dict: Dict[str, np.ndarray], path: str):
    """Save prepared GNN data as .npz for Colab upload."""
    np.savez_compressed(path, **data_dict)
    print(f"Saved GNN data to {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


def load_gnn_data(path: str) -> Dict[str, np.ndarray]:
    """Load prepared GNN data from .npz."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def numpy_to_pyg(data_dict: Dict[str, np.ndarray]) -> "Data":
    """Convert numpy data dict to PyTorch Geometric Data object."""
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError("PyTorch Geometric required. pip install torch-geometric")

    return Data(
        x=torch.tensor(data_dict["features"], dtype=torch.float),
        edge_index=torch.tensor(data_dict["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(data_dict["edge_weight"], dtype=torch.float).unsqueeze(-1),
        y=torch.tensor(data_dict["target_velocity"], dtype=torch.float),
        resistance=torch.tensor(data_dict["resistance"], dtype=torch.float),
        pos=torch.tensor(data_dict["spatial_coords"], dtype=torch.float),
    )


# ================================================================
# Inference from saved model weights
# ================================================================

def load_trained_model(
    weights_path: str,
    in_channels: int = 53,
    hidden_channels: int = 128,
    device: str = "cpu",
) -> "MechanoVelocityGNN":
    """Load a trained GNN from saved weights."""
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError("PyTorch Geometric required for GNN inference.")

    model = MechanoVelocityGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"Loaded GNN model from {weights_path}")
    return model


def predict_velocity(
    model: "MechanoVelocityGNN",
    data_dict: Dict[str, np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """Run GNN inference and return predicted velocity as numpy array."""
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError("PyTorch Geometric required for GNN inference.")

    with torch.no_grad():
        pyg_data = numpy_to_pyg(data_dict).to(device)
        model.eval()
        pred = model(pyg_data)
        return pred.cpu().numpy()

