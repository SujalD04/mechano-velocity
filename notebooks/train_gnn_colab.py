"""
Mechano-Velocity GNN Training Script (for Google Colab)

Usage in Colab:
    1. Upload gnn_training_data.npz (exported from local pipeline)
    2. Run this script
    3. Download gnn_weights.pt back to models/ folder

Or run the full notebook cell blocks below.
"""

# =============================================
# CELL 1: Setup & Imports
# =============================================
# !pip install torch-geometric torch-scatter torch-sparse -q

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================
# CELL 2: Model Definition
# =============================================

class MechanoVelocityGNN(nn.Module):
    """3-layer GCN with residual connections and physics-constrained output."""

    def __init__(self, in_channels, hidden=128, out_channels=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.norm1 = GraphNorm(hidden)
        self.norm2 = GraphNorm(hidden)
        self.norm3 = GraphNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, out_channels),
        )
        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        ew = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None

        x = F.relu(self.input_proj(x))

        r = x
        x = F.relu(self.norm1(self.conv1(x, ei, edge_weight=ew))) + r
        x = F.dropout(x, p=self.dropout, training=self.training)

        r = x
        x = F.relu(self.norm2(self.conv2(x, ei, edge_weight=ew))) + r
        x = F.dropout(x, p=self.dropout, training=self.training)

        r = x
        x = F.relu(self.norm3(self.conv3(x, ei, edge_weight=ew))) + r

        return self.mlp(x)


# =============================================
# CELL 3: Load Data
# =============================================

DATA_PATH = "gnn_training_data.npz"     # Upload this from local project
WEIGHTS_PATH = "gnn_weights.pt"          # Will be saved here

data_np = np.load(DATA_PATH)
print("Loaded arrays:", list(data_np.files))
print(f"  Features shape: {data_np['features'].shape}")
print(f"  Edge index shape: {data_np['edge_index'].shape}")
print(f"  Target velocity shape: {data_np['target_velocity'].shape}")

pyg_data = Data(
    x=torch.tensor(data_np["features"], dtype=torch.float),
    edge_index=torch.tensor(data_np["edge_index"], dtype=torch.long),
    edge_attr=torch.tensor(data_np["edge_weight"], dtype=torch.float).unsqueeze(-1),
    y=torch.tensor(data_np["target_velocity"], dtype=torch.float),
    resistance=torch.tensor(data_np["resistance"], dtype=torch.float),
    pos=torch.tensor(data_np["spatial_coords"], dtype=torch.float),
).to(device)

n_nodes = pyg_data.x.shape[0]
in_channels = pyg_data.x.shape[1]
print(f"\nGraph: {n_nodes} nodes, {pyg_data.edge_index.shape[1]} edges, {in_channels} features")

# Train / val split (80/20 random node split)
perm = torch.randperm(n_nodes)
n_train = int(0.8 * n_nodes)
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
val_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[perm[:n_train]] = True
val_mask[perm[n_train:]] = True
print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}")


# =============================================
# CELL 4: Training
# =============================================

EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_PHYSICS = 0.5  # Physics loss weight
PATIENCE = 30         # Early stopping patience

model = MechanoVelocityGNN(in_channels=in_channels, hidden=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

mse_loss = nn.MSELoss()

history = {"train_loss": [], "val_loss": [], "physics_loss": [], "lr": []}
best_val_loss = float("inf")
patience_counter = 0

print(f"\nTraining for {EPOCHS} epochs (early stopping patience={PATIENCE})")
print("=" * 60)

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    optimizer.zero_grad()

    pred = model(pyg_data)

    # Velocity MSE (train nodes only)
    loss_vel = mse_loss(pred[train_mask], pyg_data.y[train_mask])

    # Physics penalty: high R spots should have low |v|
    pred_mag_sq = (pred ** 2).sum(dim=1)
    loss_physics = (pyg_data.resistance * pred_mag_sq).mean()

    loss = loss_vel + LAMBDA_PHYSICS * loss_physics
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # --- Validate ---
    model.eval()
    with torch.no_grad():
        pred_val = model(pyg_data)
        val_loss = mse_loss(pred_val[val_mask], pyg_data.y[val_mask]).item()

    scheduler.step(val_loss)
    cur_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(loss_vel.item())
    history["val_loss"].append(val_loss)
    history["physics_loss"].append(loss_physics.item())
    history["lr"].append(cur_lr)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), WEIGHTS_PATH)
    else:
        patience_counter += 1

    if epoch % 20 == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(f"Epoch {epoch:4d} | Train: {loss_vel.item():.6f} | Val: {val_loss:.6f} | "
              f"Physics: {loss_physics.item():.6f} | LR: {cur_lr:.1e} | {elapsed:.0f}s")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (best val: {best_val_loss:.6f})")
        break

total_time = time.time() - t0
print(f"\nTraining complete in {total_time:.1f}s")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Model saved to: {WEIGHTS_PATH}")


# =============================================
# CELL 5: Training Curves
# =============================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history["train_loss"], label="Train", alpha=0.8)
axes[0].plot(history["val_loss"], label="Val", alpha=0.8)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Velocity Loss")
axes[0].legend()
axes[0].set_yscale("log")

axes[1].plot(history["physics_loss"], color="orange", alpha=0.8)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Physics Loss")
axes[1].set_title("R × ||v||² Penalty")

axes[2].plot(history["lr"], color="green", alpha=0.8)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("LR Schedule")

plt.tight_layout()
plt.savefig("gnn_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================
# CELL 6: Evaluation & Comparison
# =============================================

model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
model.eval()

with torch.no_grad():
    gnn_velocity = model(pyg_data).cpu().numpy()

target_velocity = data_np["target_velocity"]
resistance = data_np["resistance"]
coords = data_np["spatial_coords"]

# Magnitudes
gnn_mag = np.linalg.norm(gnn_velocity, axis=1)
target_mag = np.linalg.norm(target_velocity, axis=1)

# Correlation: GNN velocity vs resistance
from scipy.stats import pearsonr, ttest_ind
r_gnn, p_gnn = pearsonr(resistance, gnn_mag)
r_target, p_target = pearsonr(resistance, target_mag)

print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(f"\nResistance-Velocity Correlation:")
print(f"  Graph diffusion:  r = {r_target:.4f}, p = {p_target:.2e}")
print(f"  GNN prediction:   r = {r_gnn:.4f}, p = {p_gnn:.2e}")

# Wall vs fluid velocity (GNN)
wall = resistance > 0.8
fluid = resistance < 0.2
gnn_wall_v = gnn_mag[wall].mean()
gnn_fluid_v = gnn_mag[fluid].mean()
t_stat, p_val = ttest_ind(gnn_mag[wall], gnn_mag[fluid])

print(f"\nWall vs Fluid Velocity (GNN):")
print(f"  Wall mean:  {gnn_wall_v:.4f}")
print(f"  Fluid mean: {gnn_fluid_v:.4f}")
print(f"  t = {t_stat:.4f}, p = {p_val:.2e}")
print(f"  Result: {'✅ PASS' if gnn_wall_v < gnn_fluid_v and p_val < 0.05 else '❌ FAIL'}")

# MSE between GNN and graph diffusion
mse_val = np.mean((gnn_velocity - target_velocity) ** 2)
print(f"\nGNN vs Graph-Diffusion MSE: {mse_val:.6f}")

# Save validation results
results = {
    "gnn_resistance_correlation": {"r": float(r_gnn), "p": float(p_gnn)},
    "graph_resistance_correlation": {"r": float(r_target), "p": float(p_target)},
    "gnn_wall_velocity": float(gnn_wall_v),
    "gnn_fluid_velocity": float(gnn_fluid_v),
    "wall_vs_fluid_ttest": {"t": float(t_stat), "p": float(p_val)},
    "gnn_vs_graph_mse": float(mse_val),
    "best_val_loss": float(best_val_loss),
    "epochs_trained": len(history["train_loss"]),
    "model_params": sum(p.numel() for p in model.parameters()),
}
with open("gnn_validation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved validation results to gnn_validation_results.json")


# =============================================
# CELL 7: Visualization
# =============================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Row 1: Spatial maps
sc0 = axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=resistance, cmap="RdBu_r", s=4, alpha=0.7)
axes[0, 0].set_title("ECM Resistance Field")
plt.colorbar(sc0, ax=axes[0, 0], shrink=0.8)
axes[0, 0].invert_yaxis()

sc1 = axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=target_mag, cmap="viridis", s=4, alpha=0.7)
axes[0, 1].set_title("Graph-Diffusion Velocity")
plt.colorbar(sc1, ax=axes[0, 1], shrink=0.8)
axes[0, 1].invert_yaxis()

sc2 = axes[0, 2].scatter(coords[:, 0], coords[:, 1], c=gnn_mag, cmap="viridis", s=4, alpha=0.7)
axes[0, 2].set_title("GNN-Predicted Velocity")
plt.colorbar(sc2, ax=axes[0, 2], shrink=0.8)
axes[0, 2].invert_yaxis()

# Row 2: Scatter plots
axes[1, 0].scatter(resistance, target_mag, s=2, alpha=0.3)
axes[1, 0].set_xlabel("Resistance")
axes[1, 0].set_ylabel("Velocity")
axes[1, 0].set_title(f"Graph-Diffusion (r={r_target:.3f})")

axes[1, 1].scatter(resistance, gnn_mag, s=2, alpha=0.3, color="orange")
axes[1, 1].set_xlabel("Resistance")
axes[1, 1].set_ylabel("Velocity")
axes[1, 1].set_title(f"GNN (r={r_gnn:.3f})")

axes[1, 2].scatter(target_mag, gnn_mag, s=2, alpha=0.3, color="green")
axes[1, 2].plot([0, target_mag.max()], [0, target_mag.max()], "r--", alpha=0.5)
axes[1, 2].set_xlabel("Graph-Diffusion Velocity")
axes[1, 2].set_ylabel("GNN Velocity")
axes[1, 2].set_title(f"Agreement (MSE={mse_val:.4f})")

plt.tight_layout()
plt.savefig("gnn_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ All outputs saved. Download these files:")
print("   1. gnn_weights.pt              → models/gnn_weights.pt")
print("   2. gnn_validation_results.json  → output/gnn_validation_results.json")
print("   3. gnn_training_curves.png      → output/gnn_training_curves.png")
print("   4. gnn_comparison.png           → output/gnn_comparison.png")
