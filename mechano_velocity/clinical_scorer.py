"""
Clinical scoring algorithms for Mechano-Velocity.

Generates clinically relevant metrics:
- Metastatic Risk Score (M_risk): Outward flux of cancer cells
- Immune Exclusion Score (I_excl): T-cell accessibility
- Mechano-Therapeutic Score (MTS): Combined prognostic indicator
"""

import numpy as np
from scipy import sparse
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from anndata import AnnData

from .config import Config, default_config


@dataclass
class ClinicalReport:
    """Container for clinical scoring results."""
    # Patient/sample info
    sample_id: str
    analysis_date: str
    
    # Core scores
    metastatic_risk_score: float
    immune_exclusion_score: float
    mechano_therapeutic_score: float
    
    # Classification
    risk_category: str
    therapeutic_recommendation: str
    
    # Supporting metrics
    n_tumor_spots: int = 0
    n_tcell_spots: int = 0
    n_boundary_spots: int = 0
    mean_boundary_resistance: float = 0.0
    
    # Detailed breakdown
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sample_id': self.sample_id,
            'analysis_date': self.analysis_date,
            'scores': {
                'metastatic_risk': self.metastatic_risk_score,
                'immune_exclusion': self.immune_exclusion_score,
                'mts': self.mechano_therapeutic_score,
            },
            'classification': {
                'risk_category': self.risk_category,
                'recommendation': self.therapeutic_recommendation,
            },
            'cell_counts': {
                'tumor_spots': self.n_tumor_spots,
                'tcell_spots': self.n_tcell_spots,
                'boundary_spots': self.n_boundary_spots,
            },
            'mean_boundary_resistance': self.mean_boundary_resistance,
            'details': self.details,
        }
    
    def to_text(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "MECHANO-VELOCITY CLINICAL REPORT",
            "=" * 60,
            f"Sample ID: {self.sample_id}",
            f"Analysis Date: {self.analysis_date}",
            "",
            "-" * 60,
            "CLINICAL SCORES",
            "-" * 60,
            f"  Metastatic Risk Score:      {self.metastatic_risk_score:.4f}",
            f"  Immune Exclusion Score:     {self.immune_exclusion_score:.4f}",
            f"  Mechano-Therapeutic Score:  {self.mechano_therapeutic_score:.4f}",
            "",
            "-" * 60,
            "CLASSIFICATION",
            "-" * 60,
            f"  Risk Category: {self.risk_category}",
            f"  Therapeutic Recommendation:",
            f"    {self.therapeutic_recommendation}",
            "",
            "-" * 60,
            "TISSUE COMPOSITION",
            "-" * 60,
            f"  Tumor spots:    {self.n_tumor_spots}",
            f"  T-cell spots:   {self.n_tcell_spots}",
            f"  Boundary spots: {self.n_boundary_spots}",
            f"  Mean boundary resistance: {self.mean_boundary_resistance:.4f}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


class ClinicalScorer:
    """
    Generate clinical risk scores from Mechano-Velocity analysis.
    
    Provides three key metrics:
    
    1. Metastatic Risk Score (M_risk):
       Sum of outward velocity flux at tumor boundary
       Higher = more aggressive tumor invasion
    
    2. Immune Exclusion Score (I_excl):
       Average resistance in T-cell rich regions
       Higher = T-cells blocked by ECM
    
    3. Mechano-Therapeutic Score (MTS):
       Ratio of T-cell infiltration to cancer metastasis flux
       MTS > 2.0: "Hot" tumor (immunotherapy responsive)
       MTS < 0.5: "Cold" tumor (needs combination therapy)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize clinical scorer.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or default_config
        self.report: Optional[ClinicalReport] = None
        
    def generate_report(
        self,
        adata: AnnData,
        sample_id: Optional[str] = None,
        tumor_cluster: Optional[str] = None,
        tcell_threshold: float = 0.5,
    ) -> ClinicalReport:
        """
        Generate complete clinical report.
        
        Args:
            adata: AnnData with resistance and velocity computed.
            sample_id: Sample identifier.
            tumor_cluster: Cluster label for tumor cells, or auto-detect.
            tcell_threshold: Expression threshold for T-cell markers.
            
        Returns:
            ClinicalReport object with all scores.
        """
        print("=" * 60)
        print("GENERATING CLINICAL REPORT")
        print("=" * 60)
        
        # Auto-generate sample ID if not provided
        if sample_id is None:
            sample_id = adata.uns.get('mechano_velocity', {}).get(
                'dataset_name', 'UNKNOWN'
            )
        
        # Identify cell type regions
        print("\n1. Identifying cell type regions...")
        tumor_mask, tcell_mask, boundary_mask = self._identify_regions(
            adata, tumor_cluster, tcell_threshold
        )
        
        # Calculate metastatic risk
        print("\n2. Calculating metastatic risk...")
        m_risk = self._calculate_metastatic_risk(adata, boundary_mask)
        
        # Calculate immune exclusion
        print("\n3. Calculating immune exclusion...")
        i_excl = self._calculate_immune_exclusion(adata, tcell_mask)
        
        # Calculate MTS
        print("\n4. Calculating Mechano-Therapeutic Score...")
        mts, details = self._calculate_mts(adata, tumor_mask, tcell_mask, boundary_mask)
        
        # Classify and recommend
        risk_category, recommendation = self._classify_patient(mts, m_risk, i_excl)
        
        # Build report
        self.report = ClinicalReport(
            sample_id=sample_id,
            analysis_date=datetime.now().isoformat(),
            metastatic_risk_score=m_risk,
            immune_exclusion_score=i_excl,
            mechano_therapeutic_score=mts,
            risk_category=risk_category,
            therapeutic_recommendation=recommendation,
            n_tumor_spots=int(tumor_mask.sum()),
            n_tcell_spots=int(tcell_mask.sum()),
            n_boundary_spots=int(boundary_mask.sum()),
            mean_boundary_resistance=float(adata.obs['resistance'][boundary_mask].mean()) if boundary_mask.sum() > 0 else 0.0,
            details=details,
        )
        
        # Store in adata
        adata.uns['clinical_report'] = self.report.to_dict()
        
        print("\n" + self.report.to_text())
        
        return self.report
    
    def _identify_regions(
        self,
        adata: AnnData,
        tumor_cluster: Optional[str],
        tcell_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Identify tumor, T-cell, and boundary regions.
        
        Returns:
            Tuple of (tumor_mask, tcell_mask, boundary_mask).
        """
        n_spots = adata.n_obs
        
        # Tumor identification
        if tumor_cluster is not None and 'leiden' in adata.obs.columns:
            tumor_mask = adata.obs['leiden'] == tumor_cluster
        else:
            # Use tumor markers if available
            tumor_mask = self._identify_by_markers(
                adata, 
                self.config.genes.tumor_markers,
                threshold=tcell_threshold
            )
        
        print(f"   Tumor spots: {tumor_mask.sum()}")
        
        # T-cell identification
        tcell_mask = self._identify_by_markers(
            adata,
            self.config.genes.tcell_markers,
            threshold=tcell_threshold
        )
        print(f"   T-cell spots: {tcell_mask.sum()}")
        
        # Boundary identification (tumor spots with non-tumor neighbors)
        boundary_mask = self._identify_boundary(adata, tumor_mask)
        print(f"   Boundary spots: {boundary_mask.sum()}")
        
        # Store masks
        adata.obs['is_tumor'] = tumor_mask
        adata.obs['is_tcell'] = tcell_mask
        adata.obs['is_boundary'] = boundary_mask
        
        return tumor_mask, tcell_mask, boundary_mask
    
    def _identify_by_markers(
        self,
        adata: AnnData,
        markers: List[str],
        threshold: float
    ) -> np.ndarray:
        """Identify cells by marker expression."""
        available = [m for m in markers if m in adata.var_names]
        
        if not available:
            # Return empty mask if no markers found
            return np.zeros(adata.n_obs, dtype=bool)
        
        # Sum expression of markers
        expr = adata[:, available].X
        if sparse.issparse(expr):
            expr = expr.toarray()
        total = np.sum(expr, axis=1).flatten()
        
        # Normalize by max
        if total.max() > 0:
            total = total / total.max()
        
        return total > threshold
    
    def _identify_boundary(
        self,
        adata: AnnData,
        tumor_mask: np.ndarray
    ) -> np.ndarray:
        """Identify boundary spots (tumor spots adjacent to non-tumor)."""
        # Get adjacency matrix
        if 'spatial_connectivities' in adata.obsp:
            adj = adata.obsp['spatial_connectivities']
        else:
            # Build simple KNN adjacency
            from sklearn.neighbors import NearestNeighbors
            coords = adata.obsm['spatial']
            nbrs = NearestNeighbors(n_neighbors=7)
            nbrs.fit(coords)
            adj = nbrs.kneighbors_graph(coords)
        
        # For each tumor spot, check if any neighbor is non-tumor
        boundary_mask = np.zeros(adata.n_obs, dtype=bool)
        
        tumor_indices = np.where(tumor_mask)[0]
        for i in tumor_indices:
            neighbors = adj.getrow(i).indices
            # If any neighbor is not tumor, this is boundary
            if np.any(~tumor_mask[neighbors]):
                boundary_mask[i] = True
        
        return boundary_mask
    
    def _calculate_metastatic_risk(
        self,
        adata: AnnData,
        boundary_mask: np.ndarray
    ) -> float:
        """
        Calculate metastatic risk score.
        
        M_risk = Σ (v_corrected · n_out) for boundary spots
        
        Measures outward flux of velocity vectors at tumor boundary.
        """
        if boundary_mask.sum() == 0:
            return 0.0
        
        if 'velocity_corrected' not in adata.obsm:
            print("   Warning: Corrected velocity not found. Using zero.")
            return 0.0
        
        velocity = adata.obsm['velocity_corrected']
        coords = adata.obsm['spatial']
        
        # Get boundary spots
        boundary_indices = np.where(boundary_mask)[0]
        boundary_coords = coords[boundary_indices]
        boundary_velocity = velocity[boundary_indices]
        
        # Calculate tumor centroid
        tumor_coords = coords[adata.obs['is_tumor'].values]
        if len(tumor_coords) == 0:
            return 0.0
        centroid = tumor_coords.mean(axis=0)
        
        # Outward normal for each boundary spot
        outward = boundary_coords - centroid
        outward_norm = np.linalg.norm(outward, axis=1, keepdims=True)
        outward_norm = np.maximum(outward_norm, 1e-6)
        n_out = outward / outward_norm
        
        # Dot product of velocity with outward normal
        flux = np.sum(boundary_velocity * n_out, axis=1)
        
        # Sum positive (outward) flux
        m_risk = float(np.maximum(flux, 0).sum())
        
        print(f"   Outward flux sum: {m_risk:.4f}")
        
        return m_risk
    
    def _calculate_immune_exclusion(
        self,
        adata: AnnData,
        tcell_mask: np.ndarray
    ) -> float:
        """
        Calculate immune exclusion score.
        
        I_excl = (1/N_tcells) × Σ R_i for T-cell spots
        
        Average resistance in T-cell rich regions.
        High score = T-cells trapped in dense ECM.
        """
        if tcell_mask.sum() == 0:
            return 0.0
        
        resistance = adata.obs['resistance'].values
        i_excl = float(resistance[tcell_mask].mean())
        
        print(f"   Mean T-cell region resistance: {i_excl:.4f}")
        
        return i_excl
    
    def _calculate_mts(
        self,
        adata: AnnData,
        tumor_mask: np.ndarray,
        tcell_mask: np.ndarray,
        boundary_mask: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate Mechano-Therapeutic Score.
        
        MTS = T-cell infiltration flux / Cancer metastasis flux
        
        Interpretation:
        - MTS > 2.0: "Hot/Leaky" - T-cells entering faster than cancer spreading
        - MTS < 0.5: "Cold/Trapped" - Cancer spreading, T-cells blocked
        """
        # T-cell infiltration flux (velocity toward tumor)
        # Calculate as velocity magnitude in T-cell spots pointing toward tumor
        tcell_flux = 0.0
        if tcell_mask.sum() > 0 and 'velocity_corrected' in adata.obsm:
            velocity = adata.obsm['velocity_corrected']
            coords = adata.obsm['spatial']
            
            # Tumor centroid
            tumor_coords = coords[tumor_mask] if tumor_mask.sum() > 0 else coords
            centroid = tumor_coords.mean(axis=0)
            
            # For each T-cell spot, check if moving toward tumor
            tcell_indices = np.where(tcell_mask)[0]
            for i in tcell_indices:
                to_tumor = centroid - coords[i]
                to_tumor_norm = np.linalg.norm(to_tumor)
                if to_tumor_norm > 0:
                    to_tumor /= to_tumor_norm
                    # Positive if moving toward tumor
                    flux = np.dot(velocity[i], to_tumor)
                    tcell_flux += max(flux, 0)
        
        # Cancer metastasis flux (from boundary calculation)
        metastasis_flux = self._calculate_metastatic_risk(adata, boundary_mask)
        
        # Avoid division by zero
        if metastasis_flux < 1e-6:
            mts = 10.0 if tcell_flux > 0 else 1.0
        else:
            mts = tcell_flux / metastasis_flux
        
        details = {
            'tcell_infiltration_flux': float(tcell_flux),
            'metastasis_flux': float(metastasis_flux),
        }
        
        print(f"   T-cell infiltration flux: {tcell_flux:.4f}")
        print(f"   Cancer metastasis flux: {metastasis_flux:.4f}")
        print(f"   MTS = {mts:.4f}")
        
        return mts, details
    
    def _classify_patient(
        self,
        mts: float,
        m_risk: float,
        i_excl: float
    ) -> Tuple[str, str]:
        """
        Classify patient and generate therapeutic recommendation.
        
        Returns:
            Tuple of (risk_category, recommendation).
        """
        if mts > 2.0:
            category = "HOT / IMMUNOTHERAPY-RESPONSIVE"
            recommendation = (
                "Standard immunotherapy (anti-PD1/PDL1) is likely to be effective. "
                "T-cells show good infiltration potential. "
                "Low ECM barrier to immune access."
            )
        elif mts > 0.5:
            category = "INTERMEDIATE / BORDERLINE"
            recommendation = (
                "Consider combination therapy. "
                "Immunotherapy may be partially effective. "
                "Monitor for ECM-mediated resistance."
            )
        else:
            category = "COLD / COMBINATION THERAPY RECOMMENDED"
            if i_excl > 0.6:
                recommendation = (
                    "Anti-fibrotic pre-treatment recommended before immunotherapy. "
                    f"High immune exclusion score ({i_excl:.2f}) indicates ECM barrier. "
                    "Consider LOX inhibitors or collagenase to improve T-cell access."
                )
            else:
                recommendation = (
                    "Combination therapy recommended. "
                    "Consider chemotherapy + immunotherapy. "
                    "Low T-cell activity requires additional intervention."
                )
        
        return category, recommendation
    
    def save_report(
        self,
        path: str,
        format: str = 'txt'
    ) -> None:
        """
        Save clinical report to file.
        
        Args:
            path: Output file path.
            format: 'txt' or 'json'.
        """
        if self.report is None:
            raise ValueError("No report generated. Call generate_report first.")
        
        import json
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'txt':
            with open(path, 'w') as f:
                f.write(self.report.to_text())
        elif format == 'json':
            with open(path, 'w') as f:
                json.dump(self.report.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Report saved to: {path}")


def generate_clinical_report(
    adata: AnnData,
    config: Optional[Config] = None,
    **kwargs
) -> ClinicalReport:
    """
    Convenience function for clinical report generation.
    
    Args:
        adata: AnnData object.
        config: Configuration object.
        **kwargs: Additional arguments.
        
    Returns:
        ClinicalReport object.
    """
    scorer = ClinicalScorer(config)
    return scorer.generate_report(adata, **kwargs)
