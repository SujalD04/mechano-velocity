"""
Export GNN training data from the existing Colab notebook outputs.

Run this locally to create gnn_training_data.npz from the adata
that was already processed in the Colab notebook.
Add this as a cell in your existing mechano_velocity.ipynb notebook.
"""

# =============================================
# ADD THIS CELL to mechano_velocity.ipynb
# (after Stage 3 - Graph & Velocity is complete)
# =============================================

import numpy as np
from pathlib import Path

# At this point, adata should have:
#   adata.obsm['X_pca']              - PCA embeddings
#   adata.obs['resistance']          - Resistance values
#   adata.obsm['spatial']            - Spatial coordinates
#   adata.obsp['spatial_connectivities'] - Graph adjacency
#   adata.obsm['velocity_corrected'] - Target velocity vectors

print("Preparing GNN training data...")

# Node features: PCA (50d) + resistance (1d) + normalized coords (2d) = 53 features
pca = adata.obsm["X_pca"][:, :50]
resistance = adata.obs["resistance"].values.reshape(-1, 1)
coords = adata.obsm["spatial"].copy().astype(np.float32)
coords_norm = coords.copy()
coords_norm -= coords_norm.min(axis=0)
coords_norm /= coords_norm.max(axis=0) + 1e-8

features = np.hstack([pca, resistance, coords_norm]).astype(np.float32)
print(f"  Features: {features.shape}")

# Edge index from spatial graph
from scipy import sparse
adj = adata.obsp["spatial_connectivities"]
rows, cols = adj.nonzero()
edge_index = np.vstack([rows, cols]).astype(np.int64)
edge_weight = np.array(adj[rows, cols]).flatten().astype(np.float32)
print(f"  Edges: {edge_index.shape[1]}")

# Target velocity (from graph diffusion)
target_velocity = adata.obsm["velocity_corrected"].astype(np.float32)
print(f"  Target velocity: {target_velocity.shape}")

# Save
out_path = "gnn_training_data.npz"
np.savez_compressed(
    out_path,
    features=features,
    edge_index=edge_index,
    edge_weight=edge_weight,
    resistance=resistance.flatten().astype(np.float32),
    target_velocity=target_velocity,
    spatial_coords=coords.astype(np.float32),
)

import os
size_mb = os.path.getsize(out_path) / 1e6
print(f"\n✅ Saved: {out_path} ({size_mb:.1f} MB)")
print("Upload this file to Colab, then run train_gnn_colab.py")
