"""
Graph construction for spatial transcriptomics using PyTorch Geometric.

Builds a hexagonal grid graph from Visium spots with resistance-weighted edges.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
from anndata import AnnData

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    Data = None

from .config import Config, default_config


@dataclass
class GraphMetrics:
    """Container for graph quality metrics."""
    n_nodes: int
    n_edges: int
    avg_degree: float
    connectivity: float
    avg_edge_weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'avg_degree': self.avg_degree,
            'connectivity': self.connectivity,
            'avg_edge_weight': self.avg_edge_weight,
        }


class GraphBuilder:
    """
    Construct spatial graphs from Visium data with resistance-weighted edges.
    
    The graph structure:
    - Nodes: Spatial spots
    - Edges: Connections between adjacent spots (hexagonal grid)
    - Edge weights: W_ij = Similarity(i,j) × (1 - R_j)
    
    Supports both:
    - Sparse adjacency matrices (for CPU/numpy operations)
    - PyTorch Geometric Data objects (for GPU/GNN training)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize graph builder.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or default_config
        self.adjacency: Optional[sparse.csr_matrix] = None
        self.edge_weights: Optional[np.ndarray] = None
        self.metrics: Optional[GraphMetrics] = None
        
    def build_spatial_graph(
        self,
        adata: AnnData,
        method: str = 'knn',
        k_neighbors: int = 6,
        include_resistance: bool = True,
        include_similarity: bool = True
    ) -> sparse.csr_matrix:
        """
        Build the spatial adjacency graph.
        
        Args:
            adata: AnnData object with spatial coordinates.
            method: Graph construction method ('knn', 'delaunay', 'radius').
            k_neighbors: Number of neighbors for KNN method.
            include_resistance: Whether to weight by resistance.
            include_similarity: Whether to weight by expression similarity.
            
        Returns:
            Sparse adjacency matrix with edge weights.
        """
        print("=" * 50)
        print("BUILDING SPATIAL GRAPH")
        print("=" * 50)
        
        # Get spatial coordinates
        coords = adata.obsm['spatial']
        n_spots = len(coords)
        
        print(f"\n1. Building topology ({method})...")
        
        # Build base adjacency
        if method == 'knn':
            adjacency = self._build_knn_graph(coords, k_neighbors)
        elif method == 'delaunay':
            adjacency = self._build_delaunay_graph(coords)
        elif method == 'radius':
            adjacency = self._build_radius_graph(coords)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"   Base graph: {adjacency.nnz} edges")
        
        # Calculate edge weights
        print("\n2. Computing edge weights...")
        weights = np.ones(adjacency.nnz)
        
        if include_similarity:
            print("   - Adding expression similarity...")
            sim_weights = self._compute_similarity_weights(adata, adjacency)
            weights = weights * sim_weights
        
        if include_resistance and 'resistance' in adata.obs.columns:
            print("   - Adding resistance penalty...")
            res_weights = self._compute_resistance_weights(adata, adjacency)
            weights = weights * res_weights
        
        # Apply weights to adjacency
        adjacency.data = weights
        self.adjacency = adjacency
        self.edge_weights = weights
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(adjacency)
        self._print_summary()
        
        # Store in adata
        adata.obsp['spatial_connectivities'] = adjacency
        adata.uns['spatial_graph'] = {
            'method': method,
            'k_neighbors': k_neighbors,
            'include_resistance': include_resistance,
            'include_similarity': include_similarity,
            'metrics': self.metrics.to_dict(),
        }
        
        return adjacency
    
    def _build_knn_graph(
        self, 
        coords: np.ndarray, 
        k: int
    ) -> sparse.csr_matrix:
        """Build K-nearest neighbors graph."""
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        n = len(coords)
        rows = np.repeat(np.arange(n), k)
        cols = indices[:, 1:].flatten()  # Exclude self
        data = np.ones(len(rows))
        
        # Make symmetric
        adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        adjacency = adjacency + adjacency.T
        adjacency.data = np.ones(adjacency.nnz)  # Reset to binary
        
        return adjacency
    
    def _build_delaunay_graph(self, coords: np.ndarray) -> sparse.csr_matrix:
        """Build Delaunay triangulation graph."""
        tri = Delaunay(coords)
        
        # Extract edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        
        edges = np.array(list(edges))
        n = len(coords)
        
        # Create symmetric adjacency
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(rows))
        
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    def _build_radius_graph(
        self, 
        coords: np.ndarray,
        radius: Optional[float] = None
    ) -> sparse.csr_matrix:
        """Build radius-based graph."""
        if radius is None:
            # Use 1.5x the median nearest neighbor distance
            nbrs = NearestNeighbors(n_neighbors=2)
            nbrs.fit(coords)
            distances, _ = nbrs.kneighbors(coords)
            radius = 1.5 * np.median(distances[:, 1])
        
        nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree')
        nbrs.fit(coords)
        adjacency = nbrs.radius_neighbors_graph(coords, mode='connectivity')
        
        # Remove self-loops
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        
        return adjacency
    
    def _compute_similarity_weights(
        self,
        adata: AnnData,
        adjacency: sparse.csr_matrix
    ) -> np.ndarray:
        """
        Compute expression similarity weights for edges.
        
        Uses cosine similarity between connected spots.
        """
        # Get expression matrix
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        
        # Get edge pairs
        rows, cols = adjacency.nonzero()
        
        # Compute similarity for each edge
        similarities = np.array([
            cosine_similarity(X[i:i+1], X[j:j+1])[0, 0]
            for i, j in zip(rows, cols)
        ])
        
        # Normalize to [0, 1] and handle negatives
        similarities = np.clip(similarities, 0, 1)
        
        print(f"     Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
        
        return similarities
    
    def _compute_resistance_weights(
        self,
        adata: AnnData,
        adjacency: sparse.csr_matrix
    ) -> np.ndarray:
        """
        Compute resistance penalty weights for edges.
        
        W_ij = (1 - R_j) where R_j is the resistance of the destination spot.
        """
        resistance = adata.obs['resistance'].values
        
        # Get edge pairs (row -> col)
        rows, cols = adjacency.nonzero()
        
        # Weight by (1 - destination resistance)
        # If destination has high resistance (wall), weight is low
        weights = 1 - resistance[cols]
        
        print(f"     Resistance weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        
        return weights
    
    def _calculate_metrics(self, adjacency: sparse.csr_matrix) -> GraphMetrics:
        """Calculate graph quality metrics."""
        n_nodes = adjacency.shape[0]
        n_edges = adjacency.nnz
        avg_degree = n_edges / n_nodes
        
        # Connectivity (fraction of nodes in largest component)
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(adjacency, directed=False)
        largest_component = np.bincount(labels).max()
        connectivity = largest_component / n_nodes
        
        # Average edge weight
        avg_edge_weight = adjacency.data.mean() if adjacency.nnz > 0 else 0.0
        
        return GraphMetrics(
            n_nodes=n_nodes,
            n_edges=n_edges,
            avg_degree=avg_degree,
            connectivity=connectivity,
            avg_edge_weight=avg_edge_weight,
        )
    
    def _print_summary(self) -> None:
        """Print graph summary."""
        print("\n" + "=" * 50)
        print("Graph Construction Complete!")
        print("=" * 50)
        if self.metrics:
            print(f"Nodes: {self.metrics.n_nodes}")
            print(f"Edges: {self.metrics.n_edges}")
            print(f"Avg degree: {self.metrics.avg_degree:.2f}")
            print(f"Connectivity: {self.metrics.connectivity:.2%}")
            print(f"Avg edge weight: {self.metrics.avg_edge_weight:.3f}")
    
    def to_pytorch_geometric(
        self,
        adata: AnnData,
        node_features: Optional[str] = None
    ) -> 'Data':
        """
        Convert to PyTorch Geometric Data object.
        
        Args:
            adata: AnnData object with computed graph.
            node_features: Key in adata.obsm for node features, or None for expression.
            
        Returns:
            PyTorch Geometric Data object.
            
        Raises:
            ImportError: If PyTorch Geometric is not installed.
        """
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "PyTorch Geometric is required for this method. "
                "Install with: pip install torch-geometric"
            )
        
        print("\nConverting to PyTorch Geometric format...")
        
        # Get adjacency
        if self.adjacency is None:
            if 'spatial_connectivities' in adata.obsp:
                self.adjacency = adata.obsp['spatial_connectivities']
            else:
                raise ValueError("No graph built. Call build_spatial_graph first.")
        
        # Edge index
        rows, cols = self.adjacency.nonzero()
        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
        
        # Edge weights
        edge_weight = torch.tensor(self.adjacency.data, dtype=torch.float)
        
        # Node features
        if node_features and node_features in adata.obsm:
            x = torch.tensor(adata.obsm[node_features], dtype=torch.float)
        else:
            # Use expression matrix
            X = adata.X
            if sparse.issparse(X):
                X = X.toarray()
            x = torch.tensor(X, dtype=torch.float)
        
        # Spatial positions
        pos = torch.tensor(adata.obsm['spatial'], dtype=torch.float)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(-1),
            pos=pos,
        )
        
        # Add additional attributes
        if 'resistance' in adata.obs.columns:
            data.resistance = torch.tensor(
                adata.obs['resistance'].values, 
                dtype=torch.float
            )
        
        print(f"Created PyG Data: {data}")
        
        return data
    
    def get_neighbors(
        self, 
        spot_idx: int,
        adata: Optional[AnnData] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get neighbors and edge weights for a specific spot.
        
        Args:
            spot_idx: Index of the spot.
            adata: AnnData object (optional, uses stored adjacency if not provided).
            
        Returns:
            Tuple of (neighbor indices, edge weights).
        """
        if self.adjacency is None:
            if adata is not None and 'spatial_connectivities' in adata.obsp:
                self.adjacency = adata.obsp['spatial_connectivities']
            else:
                raise ValueError("No graph available.")
        
        row = self.adjacency.getrow(spot_idx)
        neighbors = row.indices
        weights = row.data
        
        return neighbors, weights


def build_spatial_graph(
    adata: AnnData,
    config: Optional[Config] = None,
    **kwargs
) -> sparse.csr_matrix:
    """
    Convenience function for graph construction.
    
    Args:
        adata: AnnData object with spatial coordinates.
        config: Configuration object.
        **kwargs: Additional arguments passed to GraphBuilder.build_spatial_graph().
        
    Returns:
        Sparse adjacency matrix.
    """
    builder = GraphBuilder(config)
    return builder.build_spatial_graph(adata, **kwargs)
