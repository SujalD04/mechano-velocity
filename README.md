# Mechano-Velocity

**Physics-Informed Graph Neural Network for Correcting Cell Migration Predictions in Spatial Transcriptomics**

## Overview

Mechano-Velocity is a computational framework that corrects "False Positive" migration predictions in spatial transcriptomics by detecting physical barriers (ECM/collagen) using gene signatures and applying resistance penalties to velocity vectors.

### The Problem
Current AI models (like scVelo) predict cell movement based on Gene Expression Kinetics (RNA splicing). They assume the tissue is empty space. If a cell expresses migration genes, these models assume it moves.

### The Reality
Solid tumors contain dense "Desmoplastic" regions (Scar Tissue). A cell may express migration genes, but if it is surrounded by a dense collagen wall, its velocity is physically zero.

### The Solution
This project builds a Physics-Informed Graph Neural Network that:
1. Detects physical barriers using gene signatures (COL1A1, LOX, MMP9, etc.)
2. Applies a "Resistance Penalty" to velocity vectors
3. Generates clinical risk scores (Metastatic Risk, Immune Exclusion, MTS)

## Installation

### Local Installation (for inference)

```bash
# Clone the repository
git clone https://github.com/your-username/mechano-velocity.git
cd mechano-velocity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install package
pip install -e .
```

### Google Colab (for training)

```python
# In Colab notebook
!git clone https://github.com/your-username/mechano-velocity.git
%cd mechano-velocity
!pip install -r requirements-colab.txt
```

## Dataset Setup

1. **Download the 10x Genomics Breast Cancer Dataset:**
   - Visit: https://www.10xgenomics.com/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0
   - Download the "Feature / cell matrix HDF5 (filtered)" file
   - Download the "Spatial imaging data" file

2. **Place files in the data directory:**
   ```
   data/
   └── V1_Breast_Cancer_Block_A/
       ├── filtered_feature_bc_matrix.h5
       └── spatial/
           ├── tissue_hires_image.png
           ├── tissue_lowres_image.png
           ├── tissue_positions_list.csv
           └── scalefactors_json.json
   ```

## Quick Start

### Python API

```python
from mechano_velocity import (
    DataLoader,
    Preprocessor,
    Mechanotyper,
    GraphBuilder,
    VelocityCorrector,
    ClinicalScorer,
    Visualizer,
    Config
)

# Load configuration
config = Config()

# Load data
loader = DataLoader(config)
adata = loader.load_visium()

# Preprocess
preprocessor = Preprocessor(config)
adata = preprocessor.run(adata)

# Calculate resistance field
mechanotyper = Mechanotyper(config)
resistance = mechanotyper.calculate_resistance(adata)

# Build spatial graph
graph_builder = GraphBuilder(config)
adjacency = graph_builder.build_spatial_graph(adata)

# Correct velocity
corrector = VelocityCorrector(config)
corrected_velocity = corrector.apply_resistance_correction(adata, graph_builder)

# Generate clinical report
scorer = ClinicalScorer(config)
report = scorer.generate_report(adata)

# Visualize
viz = Visualizer(config)
viz.plot_comparison(adata, save_path="output/analysis.png")
```

### Colab Notebooks

1. **01_Preprocessing.ipynb** - Data loading and quality control
2. **02_Mechanotyping.ipynb** - Resistance field calculation
3. **03_Graph_Simulation.ipynb** - Velocity correction
4. **04_Training_Validation.ipynb** - Model training and validation

## Project Structure

```
mechano-velocity/
├── mechano_velocity/           # Core Python package
│   ├── __init__.py
│   ├── config.py              # Hyperparameters & constants
│   ├── data_loader.py         # Visium data loading
│   ├── preprocessor.py        # Filtering, normalization
│   ├── mechanotyper.py        # Resistance field calculation
│   ├── graph_builder.py       # PyTorch Geometric graphs
│   ├── velocity_corrector.py  # Physics-constrained velocity
│   ├── clinical_scorer.py     # MTS, risk scores
│   ├── visualizer.py          # Plotting utilities
│   └── database.py            # Output storage
│
├── notebooks/                  # Google Colab notebooks
├── data/                       # Dataset folder (create this)
├── output/                     # Generated outputs
├── models/                     # Trained model checkpoints
└── tests/                      # Unit tests
```

## Core Equations

### Resistance Field
```
D_i = (α × COL1A1 + α × COL1A2) × (1 + β × LOX) - (γ × MMP9)
R_i = sigmoid(D_i - μ)
```

### Edge Weights
```
W_ij = Similarity(i, j) × (1 - R_j)
```

### Corrected Velocity
```
v_corrected[i] = Σ W_ij × (x_j - x_i)
```

### Mechano-Therapeutic Score (MTS)
```
MTS = T-cell_infiltration_flux / Cancer_metastasis_flux
```
- MTS > 2.0: "Hot" tumor (immunotherapy responsive)
- MTS < 0.5: "Cold" tumor (needs combination therapy)

## License

MIT License - see LICENSE file for details.

## Citation

If you use Mechano-Velocity in your research, please cite:
```
@software{mechano_velocity,
  author = {Your Name},
  title = {Mechano-Velocity: Physics-Informed Cell Migration Prediction},
  year = {2026},
  url = {https://github.com/your-username/mechano-velocity}
}
```
