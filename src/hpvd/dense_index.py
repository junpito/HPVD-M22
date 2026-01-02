"""
Dense Trajectory Index
======================

FAISS-based dense index for trajectory embeddings.
Supports exact and approximate nearest neighbor search.
"""

import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import pickle


class FAISSIndexType(Enum):
    FLAT_IP = "flat_ip"           # Exact inner product
    FLAT_L2 = "flat_l2"           # Exact L2 distance
    IVF_FLAT = "ivf_flat"         # Approximate with IVF
    HNSW = "hnsw"                  # Hierarchical NSW


@dataclass
class DenseIndexConfig:
    """Configuration for FAISS dense index"""
    
    # Basic settings
    dimension: int = 256
    index_type: FAISSIndexType = FAISSIndexType.FLAT_IP
    use_cosine: bool = True       # Normalize vectors for cosine similarity
    
    # IVF settings (if using IVF index)
    ivf_nlist: int = 100          # Number of clusters
    ivf_nprobe: int = 10          # Clusters to search
    
    # HNSW settings (if using HNSW)
    hnsw_M: int = 32              # Connections per layer
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64


class DenseTrajectoryIndex:
    """
    FAISS-based dense index for trajectory embeddings
    
    Supports:
    - Exact search (IndexFlatIP/L2)
    - Approximate search (IVF, HNSW)
    - Normalized vectors for cosine similarity
    """
    
    def __init__(self, config: DenseIndexConfig = None):
        self.config = config or DenseIndexConfig()
        self.index: Optional[faiss.Index] = None
        
        # ID mappings
        self.idx_to_id: Dict[int, str] = {}    # FAISS idx → trajectory_id
        self.id_to_idx: Dict[str, int] = {}    # trajectory_id → FAISS idx
        
        # State
        self.is_trained: bool = False
        self.ntotal: int = 0
    
    def build(self, 
              embeddings: np.ndarray,
              trajectory_ids: List[str]) -> None:
        """
        Build index from embeddings
        
        Args:
            embeddings: (N, 256) array of trajectory embeddings
            trajectory_ids: List of N trajectory IDs
        """
        n, d = embeddings.shape
        assert d == self.config.dimension, f"Expected dim {self.config.dimension}, got {d}"
        assert n == len(trajectory_ids), "Embeddings and IDs must have same length"
        
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.config.use_cosine:
            faiss.normalize_L2(embeddings)
        
        # Create index based on type
        self.index = self._create_index(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        
        # Build ID mappings
        for i, tid in enumerate(trajectory_ids):
            self.idx_to_id[i] = tid
            self.id_to_idx[tid] = i
        
        self.ntotal = n
        self.is_trained = True
        
        print(f"Built {self.config.index_type.value} index: {n} vectors, {d} dims")
    
    def _create_index(self, train_data: np.ndarray) -> faiss.Index:
        """Create FAISS index based on config"""
        d = self.config.dimension
        
        if self.config.index_type == FAISSIndexType.FLAT_IP:
            return faiss.IndexFlatIP(d)
        
        elif self.config.index_type == FAISSIndexType.FLAT_L2:
            return faiss.IndexFlatL2(d)
        
        elif self.config.index_type == FAISSIndexType.IVF_FLAT:
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.config.ivf_nlist)
            index.train(train_data)
            index.nprobe = self.config.ivf_nprobe
            return index
        
        elif self.config.index_type == FAISSIndexType.HNSW:
            index = faiss.IndexHNSWFlat(d, self.config.hnsw_M)
            index.hnsw.efConstruction = self.config.hnsw_ef_construction
            index.hnsw.efSearch = self.config.hnsw_ef_search
            return index
        
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
    
    def search(self,
               query_embedding: np.ndarray,
               k: int = 25) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors
        
        Args:
            query_embedding: (256,) query vector
            k: Number of neighbors
            
        Returns:
            List of (trajectory_id, distance) tuples, sorted by distance
        """
        if not self.is_trained:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Prepare query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        if self.config.use_cosine:
            faiss.normalize_L2(query)
        
        # Search
        k_actual = min(k, self.ntotal)
        distances, indices = self.index.search(query, k_actual)
        
        # Build results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1:
                continue
            
            tid = self.idx_to_id.get(int(idx))
            if tid is None:
                continue
            
            dist = float(distances[0][i])
            
            # Convert similarity to distance for inner product
            if self.config.index_type in [FAISSIndexType.FLAT_IP, FAISSIndexType.IVF_FLAT]:
                dist = 1.0 - dist  # cosine distance = 1 - cosine similarity
            
            results.append((tid, dist))
        
        return results
    
    def search_with_filter(self,
                           query_embedding: np.ndarray,
                           candidate_ids: Set[str],
                           k: int = 25) -> List[Tuple[str, float]]:
        """
        Search within filtered candidates
        
        Note: For MVP, we search all and filter post-hoc.
        For production, use IDSelector for efficiency.
        
        Args:
            query_embedding: (256,) query vector
            candidate_ids: Set of valid trajectory IDs
            k: Number of neighbors
            
        Returns:
            List of (trajectory_id, distance) tuples
        """
        # Search more to account for filtering
        search_k = min(k * 5, self.ntotal)
        all_results = self.search(query_embedding, search_k)
        
        # Filter
        filtered = [
            (tid, dist) for tid, dist in all_results
            if tid in candidate_ids
        ]
        
        return filtered[:k]
    
    def batch_search(self,
                     query_embeddings: np.ndarray,
                     k: int = 25) -> List[List[Tuple[str, float]]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: (M, 256) query vectors
            k: Number of neighbors per query
            
        Returns:
            List of M result lists
        """
        queries = query_embeddings.astype(np.float32)
        if self.config.use_cosine:
            faiss.normalize_L2(queries)
        
        k_actual = min(k, self.ntotal)
        distances, indices = self.index.search(queries, k_actual)
        
        results = []
        for q in range(len(queries)):
            q_results = []
            for i in range(k_actual):
                idx = indices[q][i]
                if idx != -1 and int(idx) in self.idx_to_id:
                    tid = self.idx_to_id[int(idx)]
                    dist = float(distances[q][i])
                    if self.config.index_type in [FAISSIndexType.FLAT_IP, FAISSIndexType.IVF_FLAT]:
                        dist = 1.0 - dist
                    q_results.append((tid, dist))
            results.append(q_results)
        
        return results
    
    def save(self, path: str):
        """Save index and mappings to disk"""
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save metadata
        meta = {
            'config': self.config,
            'idx_to_id': self.idx_to_id,
            'id_to_idx': self.id_to_idx,
            'ntotal': self.ntotal,
            'is_trained': self.is_trained
        }
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(meta, f)
    
    def load(self, path: str):
        """Load index and mappings from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Load metadata
        with open(f"{path}.meta", 'rb') as f:
            meta = pickle.load(f)
        
        self.config = meta['config']
        self.idx_to_id = meta['idx_to_id']
        self.id_to_idx = meta['id_to_idx']
        self.ntotal = meta['ntotal']
        self.is_trained = meta['is_trained']
        
        # Restore search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.config.ivf_nprobe
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = self.config.hnsw_ef_search

