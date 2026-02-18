# Mechano-Velocity — Progress Report

**Project:** Physics-Informed Cell Migration Prediction for Spatial Transcriptomics  
**Date:** February 2026  
**Status:** ~50% Implementation Complete (Stages 1–2 of 4)

---

## 1. What is This Project?

### The Problem

Current AI tools like **scVelo** predict where cells will migrate based on their RNA expression (a process called **RNA Velocity**). However, these tools treat the tissue as **empty space** — if a cell expresses migration genes, the model assumes it moves freely.

**In reality**, solid tumors (like breast cancer) contain dense **collagen walls** — scar-like tissue called the **desmoplastic stroma**. A cell may *want* to move, but if it's surrounded by a collagen barricade, it physically *cannot*. Current models produce **false-positive migration predictions** because they ignore this.

### Our Solution

**Mechano-Velocity** is a computational framework that:

1. **Detects physical barriers** using gene expression signatures (collagen, enzymes)
2. **Assigns a resistance score** to each tissue spot — how "blocked" it is
3. **Corrects velocity predictions** — cells in walls get their predicted movement reduced
4. **Generates clinical scores** — predicting whether immunotherapy will work based on the physical landscape

---

## 2. Dataset

We use the **10x Genomics Visium** spatial transcriptomics dataset:

| Property | Value |
|---|---|
| Tissue | Human Breast Cancer (Invasive Ductal Carcinoma) |
| Sample | Block A, Section 1 |
| Technology | 10x Visium (spatial gene expression) |
| Why this dataset? | Contains visible fibrotic (collagen-rich) regions, perfect for testing our model |

Each "spot" on the tissue slide captures the gene expression of ~1-10 cells, along with their **spatial coordinates** on the tissue.

---

## 3. Notebook 01 — Preprocessing

**Goal:** Load raw data, clean it, and prepare it for analysis.

### What Happens Step by Step

#### 3.1 Environment & Data Loading
- Detect if running in **Google Colab** or locally
- Download the 10x Genomics Breast Cancer dataset (gene expression matrix + spatial images)
- Load the data into an **AnnData** object using `scanpy` — the standard data structure for single-cell genomics
- Visualize the H&E-stained tissue image (this is the actual microscopy photo of the tissue slice)

#### 3.2 Quality Control (QC)
We filter out low-quality spots using three metrics:

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **Total UMI counts** | How many RNA molecules were captured | Too few = empty spot or dead cells |
| **Genes detected** | How many unique genes are expressed | Too few = poor capture quality |
| **Mitochondrial %** | Fraction of RNA from mitochondria | High % = dying/stressed cells (leaky membranes) |

These are visualized as histograms and spatial heatmaps, so we can see *where* on the tissue the quality is good or bad.

#### 3.3 Preprocessing Pipeline
Using our custom `Preprocessor` class, we apply:

1. **Spot filtering** — Remove spots below quality thresholds
2. **CPM Normalization** — Scale each spot to the same total (Counts Per Million), so we can compare across spots
3. **Log transformation** — `log1p(x)` to compress the dynamic range (a gene with 10,000 counts vs 10 is better compared as 9.2 vs 2.3)
4. **Highly Variable Genes (HVG)** — Select the ~2,000 most informative genes (out of ~30,000) to reduce noise
5. **PCA** — Reduce dimensionality from ~2,000 genes to ~50 principal components
6. **Neighbor graph** — Connect each spot to its most similar neighbors in gene-expression space

#### 3.4 Clustering & Gene Verification
- Run **UMAP** (for 2D visualization) and **Leiden clustering** to group spots into biologically meaningful clusters
- Verify that the key **mechanotyping genes** (COL1A1, LOX, MMP9, etc.) are present in the dataset
- Visualize these genes spatially — we can already see collagen-rich regions lighting up on the tissue
- **Save** the processed AnnData object for the next notebook

#### 3.5 Key Output
A cleaned, normalized, clustered dataset ready for mechanotyping. Saved as `preprocessed_adata.h5ad`.

---

## 4. Notebook 02 — Mechanotyping

**Goal:** Calculate the **resistance field** — a map of where physical barriers exist in the tissue.

This is the **core scientific contribution** of the project.

### 4.1 The Biology: Three Teams of Players

| Team | Role | Key Genes | Analogy |
|---|---|---|---|
| 🧱 **Construction** | Build the collagen wall | COL1A1, COL1A2, LOX | Bricks + Cement |
| 🔨 **Demolition** | Break down the wall | MMP9 | Drill / Wrecking ball |
| 🏃 **Travelers** | Cells trying to move | CD8A (T-cells), tumor markers | People trying to walk through |

- **COL1A1 / COL1A2** — Collagen type I genes. These are the **bricks** of the wall. High expression = lots of structural protein being deposited.
- **LOX** — Lysyl Oxidase. This enzyme **cross-links** collagen fibers, making the wall stiffer and harder to penetrate. Think of it as the **cement** that hardens the bricks together.
- **MMP9** — Matrix Metalloproteinase 9. This enzyme **degrades** the extracellular matrix, creating tunnels. It opposes the wall-builders.

### 4.2 The Physics Equation

The **raw density** score for each tissue spot is calculated as:

$$D_i = (\alpha \cdot COL1A1 + \alpha \cdot COL1A2) \times (1 + \beta \cdot LOX) - (\gamma \cdot MMP9)$$

Where:
- **α** (alpha) = weight for collagen (default 1.0). More collagen → higher density.
- **β** (beta) = LOX multiplier (default 0.5). LOX amplifies the collagen effect — it doesn't build walls alone, but it *hardens* existing collagen.
- **γ** (gamma) = MMP penalty (default 0.8). MMP9 degrades the wall, so it *reduces* density.

The raw density is then **normalized to [0, 1]** using a sigmoid function:

$$R_i = \frac{1}{1 + e^{-(D_i - \mu)}}$$

Where **μ** is the dataset mean density (centers the sigmoid). This converts the arbitrary density score into a **probability of being a wall** between 0 and 1.

### 4.3 KNN Smoothing

Real spatial transcriptomics data suffers from **zero-inflation** — many genes read as zero even when they're actually expressed, due to capture inefficiency. We apply **K-Nearest Neighbor smoothing**: for each spot, we average the gene values with its spatial neighbors, filling in the zeros and producing a smoother, more realistic resistance map.

### 4.4 Resistance Categories

After calculating R, each spot is classified:

| Category | Condition | Meaning |
|---|---|---|
| **Wall** 🔴 | R > 0.65 | Dense barrier — cells cannot pass |
| **Normal** ⚪ | 0.35 ≤ R ≤ 0.65 | Intermediate resistance |
| **Fluid** 🔵 | R < 0.35 | Open space — cells move freely |

### 4.5 Validation Against Histology

The resistance map is overlaid on the **H&E stained image** (actual microscopy photo). In H&E staining, **pink eosinophilic streaks** represent collagen fibers. Our validation checks:

- ✅ Do high-resistance (red) regions match the pink fibrous areas?
- ✅ Are low-resistance (blue) regions in loose stroma or fat tissue?
- ✅ Is the tumor core surrounded by a resistance ring?

### 4.6 Virtual Drug Simulation

We simulate a **LOX inhibitor drug** — what happens if we completely knock out LOX expression?

- **Hypothesis:** If LOX (cement) is removed, the collagen wall should weaken, reducing resistance
- **Method:** Set LOX expression to 0 and re-run the resistance equation, keeping the original sigmoid center (μ) so the comparison is fair
- **Expected Result:** Mean resistance drops, and some "wall" spots get reclassified as "normal" or "fluid"

This demonstrates that our model is **sensitive to therapeutic interventions** — a key requirement for clinical utility.

### 4.7 Correlation Analysis

Resistance values are compared against the Leiden clusters from Notebook 01. This reveals which cell populations are associated with high vs. low resistance, helping identify which clusters represent fibrotic tissue vs. tumor vs. immune-infiltrated regions.

### 4.8 Key Outputs
- **`resistance`** — per-spot resistance value [0, 1] stored in `adata.obs`
- **`resistance_category`** — wall / normal / fluid classification
- **`raw_density`** — the pre-sigmoid density score
- Saved as `mechanotyped_adata.h5ad` for the next notebook

---

## 5. What Comes Next (Not Yet Presented)

| Stage | Notebook | Description |
|---|---|---|
| **Graph Simulation** | `03_Graph_Simulation` | Build a spatial graph where edge weights incorporate resistance. Apply physics-constrained velocity correction — cells in walls have their velocity reduced. |
| **Training & Validation** | `04_Training_Validation` | Generate clinical scores (Mechano-Therapeutic Score), classify tumor as Hot/Cold, save full clinical report. |

---

## 6. Code Architecture

The project is built as a modular Python package (`mechano_velocity/`) with 10 modules:

| Module | Purpose | Used in NB 01 | Used in NB 02 |
|---|---|---|---|
| `config.py` | Hyperparameters, gene signatures, thresholds | ✅ | ✅ |
| `data_loader.py` | Load 10x Visium data into AnnData | ✅ | |
| `preprocessor.py` | QC, normalization, PCA, clustering | ✅ | |
| `mechanotyper.py` | Resistance field calculation + drug simulation | | ✅ |
| `visualizer.py` | All plotting and visualization utilities | | ✅ |
| `graph_builder.py` | Spatial graph construction | | |
| `velocity_corrector.py` | Physics-constrained velocity correction | | |
| `clinical_scorer.py` | MTS + clinical risk classification | | |
| `database.py` | SQLite storage for analysis runs | | |
| `__init__.py` | Package exports | ✅ | ✅ |

---

## 7. Key Libraries & Dependencies

| Library | Purpose |
|---|---|
| `scanpy` | Core single-cell analysis framework |
| `scvelo` | RNA velocity computation |
| `numpy` / `scipy` | Numerical computation, sigmoid function, sparse matrices |
| `matplotlib` | Visualization |
| `pandas` | Data manipulation |
| `torch` + `torch_geometric` | GNN compatibility (future stages) |
| `scikit-learn` | KNN smoothing, normalization |
