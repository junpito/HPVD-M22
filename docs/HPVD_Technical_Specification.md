# HPVD Technical Specification

## Deep Dive into Hybrid Probabilistic Vector Database

**Version:** 1.0.0-MVP
**Date:** December 2025
**Project:** Matrix22
**Related:** [HPVD_Architecture_Document.md](./HPVD_Architecture_Document.md)

**Source Documents:**

- [KALIBRY FINANCIAL MVP](../sources/KALIBRY%20FINANCIAL%20MVP.docx-20251209213923.md) - Primary vision, R45 features, trajectory structure
- [HPVD + PMR-DB Sprint Plan](../sources/How%20HPVD%20+%20PMR-DB%20Power%20Kalibry%20Finance-20251209213850.md) - Integration flow, validation targets
- [HPVD Concept](../sources/HPVD-20251210145834.md) - Core HPVD principles

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Trajectory Data Model](#2-trajectory-data-model)
3. [Sparse Index Specification](#3-sparse-index-specification)
4. [Dense Index Specification](#4-dense-index-specification)
5. [Hybrid Distance Metrics](#5-hybrid-distance-metrics)
6. [Search Pipeline](#6-search-pipeline)
7. [Quality Gates &amp; Validation](#7-quality-gates--validation)
8. [Storage &amp; Persistence](#8-storage--persistence)
9. [Configuration Reference](#9-configuration-reference)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Introduction

### 1.1 Purpose of This Document

Dokumen ini memberikan spesifikasi teknis mendalam untuk HPVD (Hybrid Probabilistic Vector Database), fokus pada:

- Detail implementasi setiap komponen
- Algoritma dan formula matematis
- Data structures dan interfaces
- Edge cases dan error handling

### 1.2 HPVD dalam Konteks Matrix22

```
┌─────────────────────────────────────────────────────────────┐
│                      MATRIX22                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Embedding  │───▶│    HPVD     │───▶│   PMR-DB    │     │
│  │   Engine    │    │  ◀──────▶   │    │             │     │
│  └─────────────┘    │  THIS DOC   │    └─────────────┘     │
│                     └─────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

HPVD menerima input berupa trajectory (60×45 matrix) dan menghasilkan output berupa K analog trajectories dengan distance scores.

### 1.3 Key Design Principles

| Principle               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| **Hybrid Search** | Kombinasi sparse filtering + dense retrieval untuk efisiensi |
| **Regime-Aware**  | Pencarian mempertimbangkan market regime                     |
| **Deterministic** | Query yang sama selalu menghasilkan hasil yang sama          |
| **Scalable**      | Desain mendukung scaling dari 100K ke 10M+ trajectories      |
| **Auditable**     | Setiap hasil dapat di-trace kembali ke source data           |

---

## 2. Trajectory Data Model

### 2.1 Trajectory Structure

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import uuid

@dataclass
class Trajectory:
    """
    Core trajectory entity - 60 days × 45 features
  
    Attributes:
        trajectory_id: Unique identifier (UUID)
        asset_id: Asset ticker (e.g., "AAPL", "BTC-USD")
        end_timestamp: End date of the 60-day window
        matrix: Raw feature matrix (60, 45)
        embedding: Reduced embedding for FAISS (256,)
        label_h1: H1 outcome (+1 or -1)
        label_h5: H5 outcome (+1 or -1)
        return_h1: Actual H1 return (float)
        return_h5: Actual H5 return (float)
        trend_regime: Trend classification (-1, 0, +1)
        volatility_regime: Volatility classification (-1, 0, +1)
        structural_regime: Structure classification (-1, 0, +1)
        asset_class: Asset category
    """
    # Identity
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str = ""
    end_timestamp: datetime = field(default_factory=datetime.now)
  
    # Data
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((60, 45), dtype=np.float32))
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))
  
    # Labels
    label_h1: int = 0
    label_h5: int = 0
    return_h1: float = 0.0
    return_h5: float = 0.0
  
    # Regimes
    trend_regime: int = 0
    volatility_regime: int = 0
    structural_regime: int = 0
  
    # Metadata
    asset_class: str = "equity"
  
    def get_regime_tuple(self) -> Tuple[int, int, int]:
        """Get regime as tuple for indexing"""
        return (self.trend_regime, self.volatility_regime, self.structural_regime)
  
    def get_flattened_matrix(self) -> np.ndarray:
        """Get flattened matrix (2700,)"""
        return self.matrix.flatten().astype(np.float32)
  
    def validate(self) -> bool:
        """Validate trajectory data integrity"""
        if self.matrix.shape != (60, 45):
            return False
        if self.embedding.shape != (256,):
            return False
        if self.label_h1 not in [-1, 0, 1]:
            return False
        if self.label_h5 not in [-1, 0, 1]:
            return False
        if self.trend_regime not in [-1, 0, 1]:
            return False
        if self.volatility_regime not in [-1, 0, 1]:
            return False
        if self.structural_regime not in [-1, 0, 1]:
            return False
        if np.isnan(self.matrix).any():
            return False
        return True
```

### 2.2 Embedding Computation

Untuk MVP, embedding dihitung dengan simple PCA reduction:

```python
from sklearn.decomposition import PCA
import numpy as np

class EmbeddingComputer:
    """Compute 256-dim embeddings from 60×45 matrices"""
  
    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.is_fitted = False
  
    def fit(self, matrices: np.ndarray):
        """
        Fit PCA on training data
      
        Args:
            matrices: (N, 60, 45) array of trajectory matrices
        """
        # Flatten to (N, 2700)
        N = matrices.shape[0]
        flattened = matrices.reshape(N, -1)
      
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(flattened)
        self.is_fitted = True
      
        print(f"PCA fitted: explained variance = {self.pca.explained_variance_ratio_.sum():.3f}")
  
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        Transform single matrix to embedding
      
        Args:
            matrix: (60, 45) trajectory matrix
          
        Returns:
            (256,) embedding vector
        """
        if not self.is_fitted:
            raise RuntimeError("EmbeddingComputer not fitted")
      
        flattened = matrix.flatten().reshape(1, -1)
        embedding = self.pca.transform(flattened)[0]
        return embedding.astype(np.float32)
  
    def transform_batch(self, matrices: np.ndarray) -> np.ndarray:
        """
        Transform batch of matrices
      
        Args:
            matrices: (N, 60, 45) array
          
        Returns:
            (N, 256) embeddings
        """
        N = matrices.shape[0]
        flattened = matrices.reshape(N, -1)
        embeddings = self.pca.transform(flattened)
        return embeddings.astype(np.float32)
  
    def save(self, path: str):
        """Save PCA model"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.pca, f)
  
    def load(self, path: str):
        """Load PCA model"""
        import pickle
        with open(path, 'rb') as f:
            self.pca = pickle.load(f)
        self.is_fitted = True
```

### 2.3 Regime Classification

```python
class RegimeClassifier:
    """Classify trajectory into regimes"""
  
    @staticmethod
    def classify_trend(matrix: np.ndarray) -> int:
        """
        Classify trend regime based on 60d slope
      
        Args:
            matrix: (60, 45) trajectory matrix
          
        Returns:
            -1 (DOWN), 0 (SIDEWAYS), +1 (UP)
        """
        # Assume slope_60d is at column index 3 (from R45 spec)
        # Use average of last 10 days
        slope_values = matrix[-10:, 3]  # slope_60d feature
        avg_slope = np.mean(slope_values)
      
        # Thresholds (normalized values)
        if avg_slope > 0.5:
            return 1   # UP
        elif avg_slope < -0.5:
            return -1  # DOWN
        else:
            return 0   # SIDEWAYS
  
    @staticmethod
    def classify_volatility(matrix: np.ndarray) -> int:
        """
        Classify volatility regime
      
        Args:
            matrix: (60, 45) trajectory matrix
          
        Returns:
            -1 (LOW), 0 (MEDIUM), +1 (HIGH)
        """
        # Assume volatility_20d is at column index 19
        vol_values = matrix[-10:, 19]
        avg_vol = np.mean(vol_values)
      
        # Percentile-based thresholds
        if avg_vol > 1.0:    # > 84th percentile (1 std)
            return 1   # HIGH
        elif avg_vol < -1.0:  # < 16th percentile
            return -1  # LOW
        else:
            return 0   # MEDIUM
  
    @staticmethod
    def classify_structural(matrix: np.ndarray) -> int:
        """
        Classify structural regime (trend-following vs mean-reverting)
      
        Args:
            matrix: (60, 45) trajectory matrix
          
        Returns:
            -1 (MEAN_REVERT), 0 (MIXED), +1 (TREND)
        """
        # Use R² values to determine structure
        # Assume r2_20d is at column index 13
        r2_values = matrix[-10:, 13]
        avg_r2 = np.mean(r2_values)
      
        if avg_r2 > 0.6:
            return 1   # TREND (high R² = strong trend)
        elif avg_r2 < 0.3:
            return -1  # MEAN_REVERT (low R² = noisy/reverting)
        else:
            return 0   # MIXED
  
    @classmethod
    def classify(cls, matrix: np.ndarray) -> Tuple[int, int, int]:
        """
        Classify all three regimes
      
        Returns:
            (trend, volatility, structural)
        """
        return (
            cls.classify_trend(matrix),
            cls.classify_volatility(matrix),
            cls.classify_structural(matrix)
        )
```

---

## 3. Sparse Index Specification

### 3.1 Data Structure

Sparse index menggunakan inverted index untuk regime-based filtering:

```python
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, List
import pickle

class SparseRegimeIndex:
    """
    Inverted index for regime-based trajectory filtering
  
    Enables O(1) lookup of trajectories by regime combination
    """
  
    def __init__(self):
        # Primary index: regime tuple → trajectory IDs
        # Key: (trend, vol, struct)
        # Value: Set of trajectory_ids
        self.regime_index: Dict[Tuple[int, int, int], Set[str]] = defaultdict(set)
      
        # Asset index: asset_id → trajectory IDs
        self.asset_index: Dict[str, Set[str]] = defaultdict(set)
      
        # Asset class index: asset_class → trajectory IDs
        self.asset_class_index: Dict[str, Set[str]] = defaultdict(set)
      
        # Reverse lookup: trajectory_id → regime tuple
        self.trajectory_regimes: Dict[str, Tuple[int, int, int]] = {}
      
        # Statistics
        self.total_count: int = 0
  
    def add(self, 
            trajectory_id: str,
            trend: int,
            volatility: int,
            structural: int,
            asset_id: str,
            asset_class: str = "equity"):
        """
        Add trajectory to all indexes
      
        Args:
            trajectory_id: Unique trajectory identifier
            trend: Trend regime (-1, 0, +1)
            volatility: Volatility regime (-1, 0, +1)
            structural: Structural regime (-1, 0, +1)
            asset_id: Asset ticker
            asset_class: Asset category
        """
        # Validate regimes
        assert trend in [-1, 0, 1], f"Invalid trend: {trend}"
        assert volatility in [-1, 0, 1], f"Invalid volatility: {volatility}"
        assert structural in [-1, 0, 1], f"Invalid structural: {structural}"
      
        regime_key = (trend, volatility, structural)
      
        # Add to indexes
        self.regime_index[regime_key].add(trajectory_id)
        self.asset_index[asset_id].add(trajectory_id)
        self.asset_class_index[asset_class].add(trajectory_id)
        self.trajectory_regimes[trajectory_id] = regime_key
      
        self.total_count += 1
  
    def remove(self, trajectory_id: str):
        """Remove trajectory from all indexes"""
        if trajectory_id not in self.trajectory_regimes:
            return
      
        regime_key = self.trajectory_regimes[trajectory_id]
      
        # Remove from regime index
        if trajectory_id in self.regime_index[regime_key]:
            self.regime_index[regime_key].remove(trajectory_id)
      
        # Remove from other indexes
        for asset_set in self.asset_index.values():
            asset_set.discard(trajectory_id)
        for class_set in self.asset_class_index.values():
            class_set.discard(trajectory_id)
      
        del self.trajectory_regimes[trajectory_id]
        self.total_count -= 1
  
    def filter_by_regime(self,
                         trend: Optional[int] = None,
                         volatility: Optional[int] = None,
                         structural: Optional[int] = None,
                         allow_adjacent: bool = True) -> Set[str]:
        """
        Filter trajectories by regime constraints
      
        Args:
            trend: Target trend regime or None for any
            volatility: Target volatility regime or None
            structural: Target structural regime or None
            allow_adjacent: Include adjacent regimes (±1)
          
        Returns:
            Set of matching trajectory IDs
        """
        result = set()
      
        for regime_key, trajectories in self.regime_index.items():
            k_trend, k_vol, k_struct = regime_key
            match = True
          
            # Check trend
            if trend is not None:
                if allow_adjacent:
                    match = match and abs(k_trend - trend) <= 1
                else:
                    match = match and k_trend == trend
          
            # Check volatility
            if volatility is not None:
                if allow_adjacent:
                    match = match and abs(k_vol - volatility) <= 1
                else:
                    match = match and k_vol == volatility
          
            # Check structural
            if structural is not None:
                if allow_adjacent:
                    match = match and abs(k_struct - structural) <= 1
                else:
                    match = match and k_struct == structural
          
            if match:
                result.update(trajectories)
      
        return result
  
    def filter_by_asset(self, asset_ids: List[str]) -> Set[str]:
        """Filter by specific assets"""
        result = set()
        for asset_id in asset_ids:
            result.update(self.asset_index.get(asset_id, set()))
        return result
  
    def filter_by_asset_class(self, asset_classes: List[str]) -> Set[str]:
        """Filter by asset classes"""
        result = set()
        for asset_class in asset_classes:
            result.update(self.asset_class_index.get(asset_class, set()))
        return result
  
    def combined_filter(self,
                        trend: Optional[int] = None,
                        volatility: Optional[int] = None,
                        structural: Optional[int] = None,
                        asset_ids: Optional[List[str]] = None,
                        asset_classes: Optional[List[str]] = None,
                        allow_adjacent: bool = True) -> Set[str]:
        """
        Apply multiple filters with intersection
      
        Returns:
            Set of trajectory IDs matching ALL criteria
        """
        result_sets = []
      
        # Regime filter
        if any(x is not None for x in [trend, volatility, structural]):
            regime_set = self.filter_by_regime(
                trend, volatility, structural, allow_adjacent
            )
            result_sets.append(regime_set)
      
        # Asset filter
        if asset_ids:
            asset_set = self.filter_by_asset(asset_ids)
            result_sets.append(asset_set)
      
        # Asset class filter
        if asset_classes:
            class_set = self.filter_by_asset_class(asset_classes)
            result_sets.append(class_set)
      
        # Return intersection
        if not result_sets:
            return set(self.trajectory_regimes.keys())
      
        result = result_sets[0]
        for s in result_sets[1:]:
            result = result.intersection(s)
      
        return result
  
    def get_regime_match_score(self,
                                query_regime: Tuple[int, int, int],
                                candidate_id: str) -> float:
        """
        Compute regime match score (0-1)
      
        Args:
            query_regime: (trend, vol, struct) of query
            candidate_id: Trajectory ID to compare
          
        Returns:
            Score from 0 (no match) to 1 (exact match)
        """
        if candidate_id not in self.trajectory_regimes:
            return 0.0
      
        candidate_regime = self.trajectory_regimes[candidate_id]
      
        # Score per dimension: 1.0 (exact), 0.5 (adjacent), 0.0 (far)
        scores = []
        for q, c in zip(query_regime, candidate_regime):
            diff = abs(q - c)
            if diff == 0:
                scores.append(1.0)
            elif diff == 1:
                scores.append(0.5)
            else:
                scores.append(0.0)
      
        return sum(scores) / len(scores)
  
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        regime_dist = {}
        for key, trajectories in self.regime_index.items():
            regime_dist[str(key)] = len(trajectories)
      
        return {
            'total_trajectories': self.total_count,
            'unique_regimes': len(self.regime_index),
            'unique_assets': len(self.asset_index),
            'unique_asset_classes': len(self.asset_class_index),
            'regime_distribution': regime_dist,
            'largest_regime': max(regime_dist.values()) if regime_dist else 0,
            'smallest_regime': min(regime_dist.values()) if regime_dist else 0
        }
  
    def save(self, path: str):
        """Save index to disk"""
        data = {
            'regime_index': dict(self.regime_index),
            'asset_index': dict(self.asset_index),
            'asset_class_index': dict(self.asset_class_index),
            'trajectory_regimes': self.trajectory_regimes,
            'total_count': self.total_count
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
  
    def load(self, path: str):
        """Load index from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
      
        self.regime_index = defaultdict(set, data['regime_index'])
        self.asset_index = defaultdict(set, data['asset_index'])
        self.asset_class_index = defaultdict(set, data['asset_class_index'])
        self.trajectory_regimes = data['trajectory_regimes']
        self.total_count = data['total_count']
```

### 3.2 Complexity Analysis

| Operation                    | Time Complexity                 | Space Complexity           |
| ---------------------------- | ------------------------------- | -------------------------- |
| `add()`                    | O(1)                            | O(1)                       |
| `remove()`                 | O(A) where A = num assets       | O(1)                       |
| `filter_by_regime()`       | O(27) = O(1)                    | O(K) where K = result size |
| `combined_filter()`        | O(F × K) where F = num filters | O(K)                       |
| `get_regime_match_score()` | O(1)                            | O(1)                       |

### 3.3 Memory Estimation

```
Memory per trajectory ≈ 
    regime_index entry: ~100 bytes
    asset_index entry: ~50 bytes
    trajectory_regimes entry: ~80 bytes
    ─────────────────────────────
    Total: ~230 bytes per trajectory

For 100K trajectories: ~23 MB
For 10M trajectories: ~2.3 GB
```

---

## 4. Dense Index Specification

### 4.1 FAISS Configuration

```python
import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

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
  
    def __init__(self, config: DenseIndexConfig):
        self.config = config
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
  
    def add_vectors(self,
                    embeddings: np.ndarray,
                    trajectory_ids: List[str]) -> None:
        """
        Incrementally add vectors to index
      
        Note: Not supported for all index types
        """
        if not self.is_trained:
            raise RuntimeError("Index not built")
      
        embeddings = embeddings.astype(np.float32)
        if self.config.use_cosine:
            faiss.normalize_L2(embeddings)
      
        start_idx = self.ntotal
        self.index.add(embeddings)
      
        for i, tid in enumerate(trajectory_ids):
            idx = start_idx + i
            self.idx_to_id[idx] = tid
            self.id_to_idx[tid] = idx
      
        self.ntotal += len(trajectory_ids)
  
    def save(self, path: str):
        """Save index and mappings to disk"""
        import pickle
      
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
        import pickle
      
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
```

### 4.2 Memory & Performance

| Index Type   | Memory (100K vectors) | Search Time | Recall |
| ------------ | --------------------- | ----------- | ------ |
| IndexFlatIP  | ~100 MB               | ~5ms        | 100%   |
| IndexIVFFlat | ~100 MB + overhead    | ~2ms        | ~95%   |
| IndexHNSW    | ~150 MB               | ~1ms        | ~92%   |

### 4.3 Index Selection Guide

```python
def select_index_type(n_trajectories: int) -> FAISSIndexType:
    """Select appropriate index type based on scale"""
    if n_trajectories < 100_000:
        return FAISSIndexType.FLAT_IP      # Exact, simple
    elif n_trajectories < 1_000_000:
        return FAISSIndexType.IVF_FLAT     # Fast approximate
    else:
        return FAISSIndexType.HNSW         # Very large scale
```

---

## 5. Hybrid Distance Metrics

### 5.1 Distance Components

```python
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

@dataclass
class DistanceConfig:
    """Configuration for hybrid distance computation"""
  
    # Component weights (must sum to 1.0)
    weight_euclidean: float = 0.3
    weight_cosine: float = 0.4
    weight_temporal: float = 0.3
  
    # Regime penalty
    regime_penalty_weight: float = 0.2
  
    # Temporal decay
    temporal_decay: float = 0.95   # Recent days weighted more
  
    def __post_init__(self):
        total = self.weight_euclidean + self.weight_cosine + self.weight_temporal
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


class HybridDistanceCalculator:
    """
    Compute hybrid distance between trajectory matrices
  
    Components:
    1. Euclidean distance (flattened vectors)
    2. Cosine distance (angular similarity)
    3. Temporal-weighted distance (recent days weighted more)
    4. Regime penalty (mismatch penalty)
    """
  
    def __init__(self, config: DistanceConfig = None):
        self.config = config or DistanceConfig()
      
        # Precompute temporal weights for 60-day window
        self.temporal_weights = self._compute_temporal_weights(60)
  
    def _compute_temporal_weights(self, window: int) -> np.ndarray:
        """
        Compute exponential decay weights
      
        w_t = decay^(T-1-t) for t = 0..T-1
        More recent days (higher t) get higher weight
        """
        decay = self.config.temporal_decay
        weights = np.array([
            decay ** (window - 1 - t) for t in range(window)
        ])
        return weights / weights.sum()  # Normalize to sum=1
  
    def euclidean_distance(self,
                           matrix_a: np.ndarray,
                           matrix_b: np.ndarray) -> float:
        """
        Euclidean distance between flattened matrices
      
        d_euc = ||flatten(A) - flatten(B)||_2
        """
        flat_a = matrix_a.flatten()
        flat_b = matrix_b.flatten()
        return float(np.linalg.norm(flat_a - flat_b))
  
    def cosine_distance(self,
                        matrix_a: np.ndarray,
                        matrix_b: np.ndarray) -> float:
        """
        Cosine distance between flattened matrices
      
        d_cos = 1 - cos(θ) = 1 - (A·B)/(||A|| ||B||)
        """
        flat_a = matrix_a.flatten()
        flat_b = matrix_b.flatten()
      
        dot = np.dot(flat_a, flat_b)
        norm_a = np.linalg.norm(flat_a)
        norm_b = np.linalg.norm(flat_b)
      
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 1.0  # Maximum distance for zero vectors
      
        cosine_sim = dot / (norm_a * norm_b)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
      
        return float(1.0 - cosine_sim)
  
    def temporal_weighted_distance(self,
                                    matrix_a: np.ndarray,
                                    matrix_b: np.ndarray) -> float:
        """
        Time-weighted L2 distance
      
        d_temp = Σ_t w_t * ||row_t(A) - row_t(B)||_2
      
        Recent days contribute more to the distance
        """
        # Per-day L2 distance
        day_distances = np.linalg.norm(matrix_a - matrix_b, axis=1)  # (60,)
      
        # Weighted sum
        return float(np.dot(self.temporal_weights, day_distances))
  
    def regime_match_score(self,
                           regime_a: Tuple[int, int, int],
                           regime_b: Tuple[int, int, int]) -> float:
        """
        Compute regime match score
      
        Returns:
            Score from 0 (no match) to 1 (exact match)
        """
        scores = []
        for va, vb in zip(regime_a, regime_b):
            diff = abs(va - vb)
            if diff == 0:
                scores.append(1.0)    # Exact match
            elif diff == 1:
                scores.append(0.5)    # Adjacent regime
            else:
                scores.append(0.0)    # Far regime
      
        return float(np.mean(scores))
  
    def compute(self,
                matrix_a: np.ndarray,
                matrix_b: np.ndarray,
                regime_a: Tuple[int, int, int],
                regime_b: Tuple[int, int, int]) -> Tuple[float, Dict]:
        """
        Compute full hybrid distance
      
        Args:
            matrix_a: (60, 45) query trajectory matrix
            matrix_b: (60, 45) candidate trajectory matrix
            regime_a: Query regime tuple (trend, vol, struct)
            regime_b: Candidate regime tuple
          
        Returns:
            (total_distance, components_dict)
        """
        # Compute component distances
        d_euc = self.euclidean_distance(matrix_a, matrix_b)
        d_cos = self.cosine_distance(matrix_a, matrix_b)
        d_temp = self.temporal_weighted_distance(matrix_a, matrix_b)
      
        # Regime match
        regime_match = self.regime_match_score(regime_a, regime_b)
      
        # Normalize distances to [0, 1] range (approximately)
        # Based on typical value ranges for normalized features
        d_euc_norm = d_euc / (np.sqrt(2700) * 2)    # Max ~104 for unit vectors
        d_cos_norm = d_cos / 2.0                     # Max 2 for opposite vectors
        d_temp_norm = d_temp / (np.sqrt(45) * 2)    # Empirical normalization
      
        # Clamp to [0, 1]
        d_euc_norm = min(d_euc_norm, 1.0)
        d_cos_norm = min(d_cos_norm, 1.0)
        d_temp_norm = min(d_temp_norm, 1.0)
      
        # Weighted combination
        base_distance = (
            self.config.weight_euclidean * d_euc_norm +
            self.config.weight_cosine * d_cos_norm +
            self.config.weight_temporal * d_temp_norm
        )
      
        # Apply regime penalty
        regime_penalty = (1.0 - regime_match) * self.config.regime_penalty_weight
        total_distance = base_distance * (1.0 + regime_penalty)
      
        # Return components for debugging/explainability
        components = {
            'euclidean_raw': d_euc,
            'euclidean_norm': d_euc_norm,
            'cosine_raw': d_cos,
            'cosine_norm': d_cos_norm,
            'temporal_raw': d_temp,
            'temporal_norm': d_temp_norm,
            'base_distance': base_distance,
            'regime_match': regime_match,
            'regime_penalty': regime_penalty,
            'total_distance': total_distance
        }
      
        return total_distance, components
  
    def feature_level_distance(self,
                               matrix_a: np.ndarray,
                               matrix_b: np.ndarray) -> np.ndarray:
        """
        Compute per-feature distance for explainability
      
        Returns:
            (45,) array of time-weighted distance per feature
        """
        diff = np.abs(matrix_a - matrix_b)  # (60, 45)
        weighted_diff = diff * self.temporal_weights.reshape(-1, 1)  # Apply temporal weights
      
        return weighted_diff.sum(axis=0)  # Sum across time: (45,)
```

### 5.2 Distance Formula Summary

```
HYBRID DISTANCE FORMULA
═══════════════════════

Given:
  A, B ∈ ℝ^(60×45)  - trajectory matrices
  R_A, R_B ∈ {-1,0,1}³ - regime tuples

Component Distances:
  d_euc = ||vec(A) - vec(B)||₂
  d_cos = 1 - (vec(A)·vec(B))/(||vec(A)|| ||vec(B)||)
  d_temp = Σᵢ wᵢ ||Aᵢ - Bᵢ||₂   where wᵢ = 0.95^(59-i) / Σⱼ 0.95^(59-j)

Normalization:
  d̂_euc = d_euc / (√2700 × 2)
  d̂_cos = d_cos / 2
  d̂_temp = d_temp / (√45 × 2)

Regime Match:
  regime_match = mean([1 - |Rₐᵢ - Rᵦᵢ|/2 for i in 0,1,2])

Combined Distance:
  base = 0.3 × d̂_euc + 0.4 × d̂_cos + 0.3 × d̂_temp
  penalty = (1 - regime_match) × 0.2
  
  TOTAL = base × (1 + penalty)
```

---

## 6. Search Pipeline

### 6.1 Complete HPVD Engine

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import time
import numpy as np

@dataclass
class HPVDConfig:
    """Configuration for HPVD engine"""
  
    # Search parameters
    default_k: int = 25
    search_k_multiplier: int = 3    # Search k*3 then rerank
    min_candidates: int = 100        # Minimum candidates for search
  
    # Distance config
    distance_config: DistanceConfig = None
  
    # Index config
    dense_index_config: DenseIndexConfig = None
  
    # Feature flags
    enable_sparse_filter: bool = True
    enable_reranking: bool = True
  
    def __post_init__(self):
        if self.distance_config is None:
            self.distance_config = DistanceConfig()
        if self.dense_index_config is None:
            self.dense_index_config = DenseIndexConfig()


@dataclass
class AnalogResult:
    """Result from similarity search"""
    trajectory_id: str
    asset_id: str
    distance: float
    faiss_distance: float
    label_h1: int
    label_h5: int
    return_h1: float
    return_h5: float
    regime_match: float
    distance_components: Dict
    end_timestamp: Optional[str] = None  # For explainability


@dataclass
class ForecastResult:
    """Probabilistic forecast with confidence intervals"""
    p_up: float                    # P(direction = UP)
    p_down: float                  # P(direction = DOWN)
    confidence_interval: Tuple[float, float]  # [lower, upper] bounds
    entropy: float                 # Uncertainty measure


@dataclass
class SearchResult:
    """Complete search result with metadata"""
    analogs: List[AnalogResult]
    query_trajectory_id: str
    k_requested: int
    k_returned: int
    candidates_after_sparse: int
    candidates_after_dense: int
    latency_ms: float
    latency_breakdown: Dict[str, float]
  
    # Probabilistic forecasts (from analog aggregation)
    forecast_h1: Optional[ForecastResult] = None
    forecast_h5: Optional[ForecastResult] = None
  
    # Quality metrics
    aci: float = 0.0
    regime_coherence: float = 0.0
  
    # Abstention flag
    should_abstain: bool = False
    abstention_reason: str = ""


class HPVDEngine:
    """
    Hybrid Probabilistic Vector Database Engine
  
    Main entry point for trajectory similarity search
    """
  
    def __init__(self, config: HPVDConfig = None):
        self.config = config or HPVDConfig()
      
        # Indexes
        self.sparse_index: Optional[SparseRegimeIndex] = None
        self.dense_index: Optional[DenseTrajectoryIndex] = None
      
        # Trajectory storage
        self.trajectories: Dict[str, Trajectory] = {}
      
        # Distance calculator
        self.distance_calc = HybridDistanceCalculator(self.config.distance_config)
      
        # State
        self.is_built = False
  
    def build(self, trajectories: List[Trajectory]):
        """
        Build indexes from trajectory list
      
        Args:
            trajectories: List of Trajectory objects
        """
        print(f"Building HPVD with {len(trajectories)} trajectories...")
        start_time = time.time()
      
        # Validate trajectories
        valid_trajectories = [t for t in trajectories if t.validate()]
        if len(valid_trajectories) < len(trajectories):
            print(f"Warning: {len(trajectories) - len(valid_trajectories)} invalid trajectories skipped")
      
        # Store trajectories
        for traj in valid_trajectories:
            self.trajectories[traj.trajectory_id] = traj
      
        # Build sparse index
        self.sparse_index = SparseRegimeIndex()
        for traj in valid_trajectories:
            self.sparse_index.add(
                trajectory_id=traj.trajectory_id,
                trend=traj.trend_regime,
                volatility=traj.volatility_regime,
                structural=traj.structural_regime,
                asset_id=traj.asset_id,
                asset_class=traj.asset_class
            )
      
        # Build dense index
        embeddings = np.array([t.embedding for t in valid_trajectories])
        trajectory_ids = [t.trajectory_id for t in valid_trajectories]
      
        self.dense_index = DenseTrajectoryIndex(self.config.dense_index_config)
        self.dense_index.build(embeddings, trajectory_ids)
      
        self.is_built = True
      
        elapsed = time.time() - start_time
        print(f"HPVD built in {elapsed:.2f}s")
        print(f"  Sparse index: {self.sparse_index.get_statistics()['unique_regimes']} regimes")
        print(f"  Dense index: {self.dense_index.ntotal} vectors")
  
    def search(self,
               query_trajectory: Trajectory,
               k: int = None) -> SearchResult:
        """
        Find k most similar trajectories
      
        Pipeline:
        1. Sparse filtering by regime
        2. Dense FAISS search
        3. Hybrid distance reranking
      
        Args:
            query_trajectory: Query trajectory
            k: Number of results (default: config.default_k)
          
        Returns:
            SearchResult with analogs and metadata
        """
        if not self.is_built:
            raise RuntimeError("HPVD not built. Call build() first.")
      
        k = k or self.config.default_k
        latency = {}
      
        total_start = time.time()
      
        # ========== STAGE 1: Sparse Filtering ==========
        stage_start = time.time()
      
        if self.config.enable_sparse_filter:
            candidate_ids = self.sparse_index.combined_filter(
                trend=query_trajectory.trend_regime,
                volatility=query_trajectory.volatility_regime,
                structural=query_trajectory.structural_regime,
                allow_adjacent=True
            )
          
            # Fallback if too few candidates
            if len(candidate_ids) < self.config.min_candidates:
                candidate_ids = self.sparse_index.combined_filter(
                    trend=query_trajectory.trend_regime,
                    allow_adjacent=True  # Only filter by trend
                )
          
            # Ultimate fallback: all trajectories
            if len(candidate_ids) < k:
                candidate_ids = set(self.trajectories.keys())
        else:
            candidate_ids = set(self.trajectories.keys())
      
        candidates_after_sparse = len(candidate_ids)
        latency['sparse_filter_ms'] = (time.time() - stage_start) * 1000
      
        # ========== STAGE 2: Dense Retrieval ==========
        stage_start = time.time()
      
        search_k = k * self.config.search_k_multiplier
        dense_results = self.dense_index.search_with_filter(
            query_embedding=query_trajectory.embedding,
            candidate_ids=candidate_ids,
            k=search_k
        )
      
        candidates_after_dense = len(dense_results)
        latency['dense_search_ms'] = (time.time() - stage_start) * 1000
      
        # ========== STAGE 3: Hybrid Reranking ==========
        stage_start = time.time()
      
        if self.config.enable_reranking:
            reranked = []
            query_regime = query_trajectory.get_regime_tuple()
          
            for tid, faiss_dist in dense_results:
                traj = self.trajectories.get(tid)
                if traj is None:
                    continue
              
                # Compute hybrid distance
                hybrid_dist, components = self.distance_calc.compute(
                    query_trajectory.matrix,
                    traj.matrix,
                    query_regime,
                    traj.get_regime_tuple()
                )
              
                reranked.append({
                    'trajectory_id': tid,
                    'asset_id': traj.asset_id,
                    'hybrid_distance': hybrid_dist,
                    'faiss_distance': faiss_dist,
                    'label_h1': traj.label_h1,
                    'label_h5': traj.label_h5,
                    'return_h1': traj.return_h1,
                    'return_h5': traj.return_h5,
                    'regime_match': components['regime_match'],
                    'components': components
                })
          
            # Sort by hybrid distance
            reranked.sort(key=lambda x: x['hybrid_distance'])
          
        else:
            # No reranking: use FAISS distance directly
            reranked = []
            for tid, faiss_dist in dense_results:
                traj = self.trajectories.get(tid)
                if traj is None:
                    continue
              
                regime_match = self.sparse_index.get_regime_match_score(
                    query_trajectory.get_regime_tuple(), tid
                )
              
                reranked.append({
                    'trajectory_id': tid,
                    'asset_id': traj.asset_id,
                    'hybrid_distance': faiss_dist,
                    'faiss_distance': faiss_dist,
                    'label_h1': traj.label_h1,
                    'label_h5': traj.label_h5,
                    'return_h1': traj.return_h1,
                    'return_h5': traj.return_h5,
                    'regime_match': regime_match,
                    'components': {}
                })
      
        latency['reranking_ms'] = (time.time() - stage_start) * 1000
      
        # ========== STAGE 4: Format Results ==========
        stage_start = time.time()
      
        analogs = [
            AnalogResult(
                trajectory_id=r['trajectory_id'],
                asset_id=r['asset_id'],
                distance=r['hybrid_distance'],
                faiss_distance=r['faiss_distance'],
                label_h1=r['label_h1'],
                label_h5=r['label_h5'],
                return_h1=r['return_h1'],
                return_h5=r['return_h5'],
                regime_match=r['regime_match'],
                distance_components=r['components']
            )
            for r in reranked[:k]
        ]
      
        latency['formatting_ms'] = (time.time() - stage_start) * 1000
      
        # ========== STAGE 5: Compute Forecasts & Quality ==========
        stage_start = time.time()
      
        # Compute probabilistic forecasts with confidence intervals
        forecast_h1 = compute_forecast_with_ci(analogs, horizon='h1')
        forecast_h5 = compute_forecast_with_ci(analogs, horizon='h5')
      
        # Compute quality metrics
        aci = compute_aci(analogs)
        regime_coherence = compute_regime_coherence(analogs)
      
        # Check for abstention
        should_abstain = False
        abstention_reason = ""
      
        quality_config = QualityGateConfig()
        if forecast_h1.entropy > quality_config.abstention_entropy_threshold:
            should_abstain = True
            abstention_reason = f"High H1 entropy: {forecast_h1.entropy:.3f}"
        elif forecast_h5.entropy > quality_config.abstention_entropy_threshold:
            should_abstain = True
            abstention_reason = f"High H5 entropy: {forecast_h5.entropy:.3f}"
        elif aci < quality_config.min_aci:
            should_abstain = True
            abstention_reason = f"Low ACI: {aci:.3f}"
      
        latency['forecast_ms'] = (time.time() - stage_start) * 1000
        latency['total_ms'] = (time.time() - total_start) * 1000
      
        return SearchResult(
            analogs=analogs,
            query_trajectory_id=query_trajectory.trajectory_id,
            k_requested=k,
            k_returned=len(analogs),
            candidates_after_sparse=candidates_after_sparse,
            candidates_after_dense=candidates_after_dense,
            latency_ms=latency['total_ms'],
            latency_breakdown=latency,
            forecast_h1=forecast_h1,
            forecast_h5=forecast_h5,
            aci=aci,
            regime_coherence=regime_coherence,
            should_abstain=should_abstain,
            abstention_reason=abstention_reason
        )
  
    def save(self, path: str):
        """Save HPVD to disk"""
        import os
        import pickle
      
        os.makedirs(path, exist_ok=True)
      
        # Save indexes
        self.sparse_index.save(f"{path}/sparse_index.pkl")
        self.dense_index.save(f"{path}/dense_index")
      
        # Save trajectories
        with open(f"{path}/trajectories.pkl", 'wb') as f:
            pickle.dump(self.trajectories, f)
      
        # Save config
        with open(f"{path}/config.pkl", 'wb') as f:
            pickle.dump(self.config, f)
      
        print(f"HPVD saved to {path}")
  
    def load(self, path: str):
        """Load HPVD from disk"""
        import pickle
      
        # Load config
        with open(f"{path}/config.pkl", 'rb') as f:
            self.config = pickle.load(f)
      
        # Load trajectories
        with open(f"{path}/trajectories.pkl", 'rb') as f:
            self.trajectories = pickle.load(f)
      
        # Load indexes
        self.sparse_index = SparseRegimeIndex()
        self.sparse_index.load(f"{path}/sparse_index.pkl")
      
        self.dense_index = DenseTrajectoryIndex(self.config.dense_index_config)
        self.dense_index.load(f"{path}/dense_index")
      
        # Rebuild distance calculator
        self.distance_calc = HybridDistanceCalculator(self.config.distance_config)
      
        self.is_built = True
        print(f"HPVD loaded from {path}: {len(self.trajectories)} trajectories")
  
    def get_statistics(self) -> Dict:
        """Get HPVD statistics"""
        return {
            'total_trajectories': len(self.trajectories),
            'sparse_index_stats': self.sparse_index.get_statistics() if self.sparse_index else {},
            'dense_index_vectors': self.dense_index.ntotal if self.dense_index else 0,
            'is_built': self.is_built
        }
```

### 6.2 Search Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    HPVD SEARCH PIPELINE                          │
└─────────────────────────────────────────────────────────────────┘

INPUT: Query Trajectory T_q
  │
  │  trajectory_id: "query_123"
  │  embedding: [0.12, -0.34, ...]  (256-dim)
  │  regime: (trend=1, vol=0, struct=1)
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: SPARSE FILTERING                              ~2ms     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query regime: (1, 0, 1)                                        │
│  Allow adjacent: True                                            │
│                                                                  │
│  Matching regimes:                                               │
│    (1, 0, 1) → exact match                                      │
│    (1, 0, 0), (1, 1, 1), (0, 0, 1), ... → adjacent              │
│                                                                  │
│  Result: 35,000 candidate IDs                                   │
│                                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: DENSE RETRIEVAL (FAISS)                       ~15ms    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query embedding: [0.12, -0.34, ...] (normalized)               │
│  Search k: 75 (k=25 × 3)                                        │
│  Filter: candidate_ids from Stage 1                             │
│                                                                  │
│  FAISS IndexFlatIP:                                             │
│    - Compute inner product with all candidates                  │
│    - Return top 75 by similarity                                │
│                                                                  │
│  Result: [(tid_1, 0.92), (tid_2, 0.88), ..., (tid_75, 0.45)]   │
│                                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: HYBRID RERANKING                              ~5ms     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each of 75 candidates:                                     │
│                                                                  │
│    1. Load full matrix from storage                             │
│    2. Compute hybrid distance:                                  │
│       d_hybrid = 0.3×d_euc + 0.4×d_cos + 0.3×d_temp            │
│                  × (1 + 0.2×regime_penalty)                     │
│    3. Store (tid, d_hybrid, components)                         │
│                                                                  │
│  Sort by d_hybrid ascending                                     │
│  Take top 25                                                    │
│                                                                  │
│  Result: 25 AnalogResult objects                                │
│                                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: FORECAST & QUALITY                              ~2ms   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Compute weighted probability:                                  │
│    P(up) = Σᵢ wᵢ × outcomeᵢ                                    │
│    wᵢ = exp(-2.0 × distanceᵢ) / Σⱼ exp(-2.0 × distanceⱼ)       │
│                                                                  │
│  Compute confidence intervals (Wilson score)                    │
│  Compute ACI, regime coherence                                  │
│                                                                  │
│  Check abstention:                                              │
│    if entropy > 0.9 → should_abstain = True                    │
│                                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
OUTPUT: SearchResult
  │
  │  analogs: [AnalogResult_1, ..., AnalogResult_25]
  │  k_returned: 25
  │  forecast_h1: {p_up: 0.68, CI: [0.63, 0.73], entropy: 0.89}
  │  forecast_h5: {p_up: 0.71, CI: [0.65, 0.77], entropy: 0.86}
  │  aci: 0.76, regime_coherence: 0.82
  │  should_abstain: False
  │  latency_ms: 24.5
  │  latency_breakdown: {sparse: 2, dense: 15, rerank: 5, forecast: 2}
```

---

## 7. Quality Gates & Validation

### 7.1 Analog Quality Metrics

```python
import numpy as np
from typing import List

def compute_aci(analogs: List[AnalogResult]) -> float:
    """
    Compute Analog Cohesion Index
  
    High ACI = analogs are similar to each other = reliable
    Low ACI = analogs are dispersed = less reliable
  
    Formula:
        ACI = 1 - (mean_distance + std_distance) / 2
  
    Returns:
        ACI in [0, 1]
    """
    if len(analogs) < 2:
        return 1.0
  
    distances = np.array([a.distance for a in analogs])
  
    mean_dist = distances.mean()
    std_dist = distances.std()
  
    # Normalize assuming max distance ~2
    aci = 1.0 - (mean_dist + std_dist) / 2.0
    return float(np.clip(aci, 0.0, 1.0))


def compute_regime_coherence(analogs: List[AnalogResult]) -> float:
    """
    Compute average regime match across analogs
  
    Returns:
        Coherence score in [0, 1]
    """
    if not analogs:
        return 0.0
  
    scores = [a.regime_match for a in analogs]
    return float(np.mean(scores))


def compute_forecast_with_ci(analogs: List[AnalogResult], 
                              horizon: str = 'h1',
                              confidence_level: float = 0.95) -> ForecastResult:
    """
    Compute probabilistic forecast with confidence intervals
  
    Uses weighted voting based on similarity (distance-based weights):
        w_i = exp(-α * distance_i)
  
    Confidence interval computed using Wilson score interval.
  
    Args:
        analogs: List of analog results
        horizon: 'h1' or 'h5'
        confidence_level: Confidence level for interval (default 0.95)
      
    Returns:
        ForecastResult with P(up), P(down), CI, and entropy
    """
    if not analogs:
        return ForecastResult(
            p_up=0.5, p_down=0.5,
            confidence_interval=(0.0, 1.0),
            entropy=1.0
        )
  
    # Distance-based weights: w_i = exp(-α * distance_i)
    alpha = 2.0  # Decay factor
    distances = np.array([a.distance for a in analogs])
    weights = np.exp(-alpha * distances)
    weights = weights / weights.sum()  # Normalize
  
    # Get outcomes
    if horizon == 'h1':
        outcomes = np.array([1.0 if a.label_h1 == 1 else 0.0 for a in analogs])
    else:
        outcomes = np.array([1.0 if a.label_h5 == 1 else 0.0 for a in analogs])
  
    # Weighted probability
    p_up = float(np.dot(weights, outcomes))
    p_down = 1.0 - p_up
  
    # Wilson score interval for confidence bounds
    n = len(analogs)
    z = 1.96 if confidence_level == 0.95 else 1.645  # z-score
  
    denominator = 1 + z**2 / n
    center = (p_up + z**2 / (2*n)) / denominator
    spread = z * np.sqrt((p_up * p_down) / n + z**2 / (4*n**2)) / denominator
  
    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)
  
    # Entropy
    if p_up <= 0 or p_up >= 1:
        entropy = 0.0
    else:
        entropy = -p_up * np.log2(p_up) - p_down * np.log2(p_down)
  
    return ForecastResult(
        p_up=p_up,
        p_down=p_down,
        confidence_interval=(ci_lower, ci_upper),
        entropy=float(entropy)
    )


def compute_outcome_entropy(analogs: List[AnalogResult], horizon: str = 'h1') -> float:
    """
    Compute entropy of analog outcomes
  
    High entropy = mixed outcomes = uncertain
    Low entropy = consistent outcomes = confident
  
    Args:
        analogs: List of analog results
        horizon: 'h1' or 'h5'
      
    Returns:
        Entropy in [0, 1]
    """
    if not analogs:
        return 1.0
  
    if horizon == 'h1':
        outcomes = [1 if a.label_h1 == 1 else 0 for a in analogs]
    else:
        outcomes = [1 if a.label_h5 == 1 else 0 for a in analogs]
  
    p_up = np.mean(outcomes)
  
    if p_up <= 0 or p_up >= 1:
        return 0.0  # No uncertainty
  
    entropy = -p_up * np.log2(p_up) - (1 - p_up) * np.log2(1 - p_up)
    return float(entropy)


def compute_brier_score(predictions: List[float], actuals: List[int]) -> float:
    """
    Compute Brier Score for probability calibration
  
    BS = (1/N) * Σ(p_pred - y_actual)²
  
    Lower is better:
    - 0.0 = perfect predictions
    - 0.25 = random guessing (for binary)
  
    Target: Brier < 0.18 (competitive with LSTM)
  
    Args:
        predictions: List of predicted probabilities P(up)
        actuals: List of actual outcomes (1=up, 0=down)
      
    Returns:
        Brier Score
    """
    if not predictions or not actuals:
        return 1.0
  
    predictions = np.array(predictions)
    actuals = np.array(actuals)
  
    return float(np.mean((predictions - actuals) ** 2))


def compute_ece(predictions: List[float], actuals: List[int], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error
  
    ECE = Σ (|B_m| / N) * |accuracy(B_m) - confidence(B_m)|
  
    Measures how well predicted probabilities match actual frequencies.
  
    Target: ECE < 8% for HPVD, ECE < 5% for full system
  
    Args:
        predictions: List of predicted probabilities
        actuals: List of actual outcomes (1=up, 0=down)
        n_bins: Number of probability bins
      
    Returns:
        ECE in [0, 1]
    """
    if not predictions or not actuals:
        return 1.0
  
    predictions = np.array(predictions)
    actuals = np.array(actuals)
  
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
  
    for i in range(n_bins):
        in_bin = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
      
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(actuals[in_bin])
            avg_confidence_in_bin = np.mean(predictions[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
  
    return float(ece)


@dataclass
class QualityGateConfig:
    """Configuration for quality gates"""
    min_k: int = 10              # Minimum analogs required
    min_aci: float = 0.7         # Minimum ACI (source: ACI > 0.7 for 80% queries)
    min_regime_coherence: float = 0.65  # Minimum regime coherence (source: RC > 0.65)
    max_mean_distance: float = 1.0     # Maximum average distance
    max_ece: float = 0.08        # Maximum Expected Calibration Error (source: ECE < 8%)
    max_brier: float = 0.18      # Maximum Brier Score (source: Brier < 0.18)
    abstention_entropy_threshold: float = 0.9  # Abstain if entropy > threshold


def validate_search_result(result: SearchResult, 
                            config: QualityGateConfig) -> Tuple[bool, List[str], bool]:
    """
    Validate search result quality
  
    Returns:
        (passed, list_of_warnings, should_abstain)
    """
    warnings = []
    passed = True
    should_abstain = False
  
    # Check minimum k
    if result.k_returned < config.min_k:
        warnings.append(f"Too few analogs: {result.k_returned} < {config.min_k}")
        passed = False
  
    # Check ACI (target: > 0.7 for 80% of queries)
    aci = compute_aci(result.analogs)
    if aci < config.min_aci:
        warnings.append(f"Low ACI: {aci:.3f} < {config.min_aci}")
        passed = False
  
    # Check regime coherence (target: RC > 0.65)
    coherence = compute_regime_coherence(result.analogs)
    if coherence < config.min_regime_coherence:
        warnings.append(f"Low regime coherence: {coherence:.3f} < {config.min_regime_coherence}")
        passed = False
  
    # Check mean distance
    if result.analogs:
        mean_dist = np.mean([a.distance for a in result.analogs])
        if mean_dist > config.max_mean_distance:
            warnings.append(f"High mean distance: {mean_dist:.3f} > {config.max_mean_distance}")
            passed = False
  
    # Check entropy for abstention
    entropy_h1 = compute_outcome_entropy(result.analogs, 'h1')
    entropy_h5 = compute_outcome_entropy(result.analogs, 'h5')
  
    if entropy_h1 > config.abstention_entropy_threshold or entropy_h5 > config.abstention_entropy_threshold:
        warnings.append(f"High entropy (H1: {entropy_h1:.3f}, H5: {entropy_h5:.3f}) > {config.abstention_entropy_threshold}")
        should_abstain = True
  
    return passed, warnings, should_abstain
```

### 7.2 Walk-Forward Validation Methodology

Walk-forward validation ensures realistic backtesting without data leakage:

```python
class WalkForwardValidator:
    """
    Walk-forward validation for HPVD
  
    Mimics real-world usage:
    - Train on past
    - Predict future
    - Roll forward
  
    Time-based splits (no random shuffling):
    - Train: 2013-2019
    - Validation: 2020-2021
    - Test: 2022-2024
    """
  
    def __init__(self, 
                 train_end: str = "2019-12-31",
                 val_end: str = "2021-12-31"):
        self.train_end = datetime.strptime(train_end, "%Y-%m-%d")
        self.val_end = datetime.strptime(val_end, "%Y-%m-%d")
  
    def split_trajectories(self, 
                           trajectories: List[Trajectory]
                           ) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
        """
        Split trajectories chronologically
      
        Returns:
            (train, validation, test) trajectory lists
        """
        train = []
        val = []
        test = []
      
        for traj in trajectories:
            if traj.end_timestamp <= self.train_end:
                train.append(traj)
            elif traj.end_timestamp <= self.val_end:
                val.append(traj)
            else:
                test.append(traj)
      
        return train, val, test
  
    def validate(self, 
                 hpvd: HPVDEngine,
                 test_trajectories: List[Trajectory]) -> Dict:
        """
        Run walk-forward validation
      
        Key rules:
        1. Query trajectory ends at day t
        2. H1 starts at t+1, H5 ends at t+5
        3. Historical analogs must be entirely in the past
        4. No future information contamination
      
        Returns:
            Validation metrics (accuracy, ECE, Brier, etc.)
        """
        predictions_h1 = []
        actuals_h1 = []
        predictions_h5 = []
        actuals_h5 = []
      
        for query in test_trajectories:
            result = hpvd.search(query)
          
            if not result.should_abstain:
                predictions_h1.append(result.forecast_h1.p_up)
                actuals_h1.append(1 if query.label_h1 == 1 else 0)
              
                predictions_h5.append(result.forecast_h5.p_up)
                actuals_h5.append(1 if query.label_h5 == 1 else 0)
      
        # Compute metrics
        accuracy_h1 = np.mean([
            (p > 0.5) == a for p, a in zip(predictions_h1, actuals_h1)
        ]) if predictions_h1 else 0.0
      
        accuracy_h5 = np.mean([
            (p > 0.5) == a for p, a in zip(predictions_h5, actuals_h5)
        ]) if predictions_h5 else 0.0
      
        return {
            'accuracy_h1': accuracy_h1,
            'accuracy_h5': accuracy_h5,
            'brier_h1': compute_brier_score(predictions_h1, actuals_h1),
            'brier_h5': compute_brier_score(predictions_h5, actuals_h5),
            'ece_h1': compute_ece(predictions_h1, actuals_h1),
            'ece_h5': compute_ece(predictions_h5, actuals_h5),
            'coverage': len(predictions_h1) / len(test_trajectories),  # % not abstained
            'n_predictions': len(predictions_h1),
            'n_abstained': len(test_trajectories) - len(predictions_h1)
        }
```

**Validation Targets (from source docs):**

- H1 Accuracy: 58-62% (vs 50% random walk)
- H5 Accuracy: 56-60%
- Brier Score: < 0.18
- ECE: < 8%
- Coverage: > 80% (abstention < 20%)

### 7.3 Search Validation Tests

```python
def test_search_determinism(hpvd: HPVDEngine, query: Trajectory, n_runs: int = 5):
    """Verify search is deterministic"""
    results = []
    for _ in range(n_runs):
        result = hpvd.search(query, k=25)
        ids = [a.trajectory_id for a in result.analogs]
        results.append(ids)
  
    # All results should be identical
    for i in range(1, n_runs):
        assert results[i] == results[0], "Search not deterministic!"
  
    print("✓ Search determinism verified")


def test_regime_filtering(hpvd: HPVDEngine, query: Trajectory):
    """Verify regime filtering is working"""
    result = hpvd.search(query, k=25)
  
    query_regime = query.get_regime_tuple()
  
    # All analogs should have similar regime
    for analog in result.analogs:
        traj = hpvd.trajectories[analog.trajectory_id]
        cand_regime = traj.get_regime_tuple()
      
        # At least 2 out of 3 regimes should match or be adjacent
        matches = sum(abs(q - c) <= 1 for q, c in zip(query_regime, cand_regime))
        assert matches >= 2, f"Regime mismatch: query={query_regime}, candidate={cand_regime}"
  
    print("✓ Regime filtering verified")


def test_distance_ordering(hpvd: HPVDEngine, query: Trajectory):
    """Verify results are ordered by distance"""
    result = hpvd.search(query, k=25)
  
    distances = [a.distance for a in result.analogs]
  
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i-1], "Results not sorted by distance!"
  
    print("✓ Distance ordering verified")


def test_quality_metrics(hpvd: HPVDEngine, query: Trajectory):
    """Verify quality metrics meet targets"""
    result = hpvd.search(query, k=25)
  
    # ACI target: > 0.7 for 80% of queries
    assert result.aci > 0.5, f"ACI too low: {result.aci}"
  
    # Regime coherence target: > 0.65
    assert result.regime_coherence > 0.5, f"Regime coherence too low: {result.regime_coherence}"
  
    # Forecast should have valid confidence intervals
    assert result.forecast_h1 is not None
    assert result.forecast_h1.confidence_interval[0] <= result.forecast_h1.p_up
    assert result.forecast_h1.p_up <= result.forecast_h1.confidence_interval[1]
  
    print("✓ Quality metrics verified")


def test_abstention(hpvd: HPVDEngine, ambiguous_query: Trajectory):
    """Verify abstention triggers for ambiguous cases"""
    result = hpvd.search(ambiguous_query, k=25)
  
    # If entropy is high, should_abstain should be True
    if result.forecast_h1.entropy > 0.9:
        assert result.should_abstain, "Should abstain on high entropy"
        assert result.abstention_reason != "", "Abstention reason should be provided"
  
    print("✓ Abstention mechanism verified")
```

---

## 8. Storage & Persistence

### 8.1 File Structure

```
matrix22-data/
├── trajectories/
│   ├── matrices/                 # Raw 60×45 matrices (.npy)
│   │   ├── AAPL/
│   │   │   ├── 2020-01-15.npy   # Single trajectory matrix
│   │   │   ├── 2020-01-16.npy
│   │   │   └── ...
│   │   ├── MSFT/
│   │   ├── BTC-USD/
│   │   └── ...
│   │
│   └── embeddings/
│       └── all_embeddings.npy    # (N, 256) precomputed embeddings
│
├── indexes/
│   ├── sparse_index.pkl          # Pickled SparseRegimeIndex
│   ├── dense_index.faiss         # FAISS binary index
│   ├── dense_index.meta          # FAISS metadata (ID mappings)
│   └── pca_model.pkl             # PCA for embedding computation
│
├── calibration/
│   ├── isotonic_h1.pkl           # H1 calibrator
│   └── isotonic_h5.pkl           # H5 calibrator
│
└── metadata/
    ├── trajectories.db           # SQLite database
    └── config.json               # System configuration
```

### 8.2 SQLite Schema

```sql
-- Trajectory metadata table
CREATE TABLE trajectories (
    trajectory_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL,
    end_timestamp TEXT NOT NULL,
    matrix_path TEXT NOT NULL,
    label_h1 INTEGER,
    label_h5 INTEGER,
    return_h1 REAL,
    return_h5 REAL,
    trend_regime INTEGER NOT NULL,
    volatility_regime INTEGER NOT NULL,
    structural_regime INTEGER NOT NULL,
    asset_class TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX idx_asset_id ON trajectories(asset_id);
CREATE INDEX idx_end_timestamp ON trajectories(end_timestamp);
CREATE INDEX idx_regimes ON trajectories(trend_regime, volatility_regime, structural_regime);
CREATE INDEX idx_asset_class ON trajectories(asset_class);

-- Forecast audit log
CREATE TABLE forecast_log (
    log_id TEXT PRIMARY KEY,
    query_trajectory_id TEXT,
    query_asset_id TEXT NOT NULL,
    query_timestamp TEXT NOT NULL,
    p_up_h1 REAL NOT NULL,
    p_up_h5 REAL NOT NULL,
    aci REAL,
    k_used INTEGER,
    latency_ms REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_forecast_time ON forecast_log(created_at);
```

### 8.3 Storage Manager

```python
import sqlite3
import json
import os
from pathlib import Path

class StorageManager:
    """Manage HPVD data persistence"""
  
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.matrices_path = self.base_path / "trajectories" / "matrices"
        self.embeddings_path = self.base_path / "trajectories" / "embeddings"
        self.indexes_path = self.base_path / "indexes"
        self.calibration_path = self.base_path / "calibration"
        self.metadata_path = self.base_path / "metadata"
      
        self.db_path = self.metadata_path / "trajectories.db"
        self._conn = None
  
    def initialize(self):
        """Create directory structure and database"""
        for path in [self.matrices_path, self.embeddings_path, 
                     self.indexes_path, self.calibration_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
      
        self._init_database()
  
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
      
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                end_timestamp TEXT NOT NULL,
                matrix_path TEXT NOT NULL,
                label_h1 INTEGER,
                label_h5 INTEGER,
                return_h1 REAL,
                return_h5 REAL,
                trend_regime INTEGER NOT NULL,
                volatility_regime INTEGER NOT NULL,
                structural_regime INTEGER NOT NULL,
                asset_class TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
          
            CREATE INDEX IF NOT EXISTS idx_asset_id ON trajectories(asset_id);
            CREATE INDEX IF NOT EXISTS idx_regimes ON trajectories(
                trend_regime, volatility_regime, structural_regime
            );
        """)
      
        conn.commit()
        conn.close()
  
    def save_trajectory(self, trajectory: Trajectory):
        """Save trajectory to storage"""
        # Save matrix
        asset_dir = self.matrices_path / trajectory.asset_id
        asset_dir.mkdir(exist_ok=True)
      
        date_str = trajectory.end_timestamp.strftime("%Y-%m-%d")
        matrix_path = asset_dir / f"{date_str}.npy"
        np.save(str(matrix_path), trajectory.matrix)
      
        # Save to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
      
        cursor.execute("""
            INSERT OR REPLACE INTO trajectories 
            (trajectory_id, asset_id, end_timestamp, matrix_path,
             label_h1, label_h5, return_h1, return_h5,
             trend_regime, volatility_regime, structural_regime, asset_class)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trajectory.trajectory_id,
            trajectory.asset_id,
            trajectory.end_timestamp.isoformat(),
            str(matrix_path.relative_to(self.base_path)),
            trajectory.label_h1,
            trajectory.label_h5,
            trajectory.return_h1,
            trajectory.return_h5,
            trajectory.trend_regime,
            trajectory.volatility_regime,
            trajectory.structural_regime,
            trajectory.asset_class
        ))
      
        conn.commit()
        conn.close()
  
    def load_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """Load trajectory from storage"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
      
        cursor.execute("""
            SELECT * FROM trajectories WHERE trajectory_id = ?
        """, (trajectory_id,))
      
        row = cursor.fetchone()
        conn.close()
      
        if row is None:
            return None
      
        # Load matrix
        matrix_path = self.base_path / row[3]  # matrix_path column
        matrix = np.load(str(matrix_path))
      
        return Trajectory(
            trajectory_id=row[0],
            asset_id=row[1],
            end_timestamp=datetime.fromisoformat(row[2]),
            matrix=matrix,
            embedding=np.zeros(256, dtype=np.float32),  # Load separately if needed
            label_h1=row[4],
            label_h5=row[5],
            return_h1=row[6],
            return_h5=row[7],
            trend_regime=row[8],
            volatility_regime=row[9],
            structural_regime=row[10],
            asset_class=row[11]
        )
  
    def get_all_trajectory_ids(self) -> List[str]:
        """Get all trajectory IDs"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
      
        cursor.execute("SELECT trajectory_id FROM trajectories")
        ids = [row[0] for row in cursor.fetchall()]
      
        conn.close()
        return ids
  
    def get_trajectory_count(self) -> int:
        """Get total trajectory count"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
      
        cursor.execute("SELECT COUNT(*) FROM trajectories")
        count = cursor.fetchone()[0]
      
        conn.close()
        return count
```

---

## 9. Configuration Reference

### 9.1 Complete Configuration

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HPVDFullConfig:
    """Complete HPVD configuration"""
  
    # ============ Trajectory Settings ============
    trajectory_window: int = 60        # Days in trajectory
    feature_count: int = 45            # R45 features
    embedding_dim: int = 256           # Reduced embedding dimension
  
    # ============ Search Settings ============
    default_k: int = 25                # Default number of neighbors
    search_k_multiplier: int = 3       # Oversample for reranking
    min_candidates: int = 100          # Minimum sparse filter results
  
    # ============ Distance Settings ============
    weight_euclidean: float = 0.3      # Euclidean weight
    weight_cosine: float = 0.4         # Cosine weight
    weight_temporal: float = 0.3       # Temporal weight
    regime_penalty: float = 0.2        # Regime mismatch penalty
    temporal_decay: float = 0.95       # Temporal weight decay
  
    # ============ Index Settings ============
    faiss_index_type: str = "flat_ip"  # flat_ip, ivf_flat, hnsw
    faiss_ivf_nlist: int = 100         # IVF clusters
    faiss_ivf_nprobe: int = 10         # IVF search clusters
    faiss_hnsw_M: int = 32             # HNSW connections
    faiss_hnsw_ef: int = 64            # HNSW search effort
  
    # ============ Quality Gates ============
    min_aci: float = 0.5               # Minimum ACI threshold
    min_regime_coherence: float = 0.5  # Minimum regime coherence
    max_mean_distance: float = 1.0     # Maximum mean distance
  
    # ============ Feature Flags ============
    enable_sparse_filter: bool = True
    enable_reranking: bool = True
    enable_quality_gates: bool = True
  
    # ============ Storage Settings ============
    data_path: str = "matrix22-data"
    use_memory_mapping: bool = False   # Memory-map large files
  
    @classmethod
    def from_json(cls, path: str) -> 'HPVDFullConfig':
        """Load config from JSON file"""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
  
    def to_json(self, path: str):
        """Save config to JSON file"""
        import json
        from dataclasses import asdict
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
```

### 9.2 Environment Variables

```bash
# .env file

# Storage
MATRIX22_DATA_PATH=./matrix22-data
MATRIX22_LOG_LEVEL=INFO

# HPVD Search
HPVD_DEFAULT_K=25
HPVD_SEARCH_MULTIPLIER=3
HPVD_MIN_CANDIDATES=100

# Distance Weights
HPVD_WEIGHT_EUCLIDEAN=0.3
HPVD_WEIGHT_COSINE=0.4
HPVD_WEIGHT_TEMPORAL=0.3
HPVD_REGIME_PENALTY=0.2

# Quality Gates
HPVD_MIN_ACI=0.5
HPVD_MIN_REGIME_COHERENCE=0.5

# Index
HPVD_INDEX_TYPE=flat_ip
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
import pytest
import numpy as np

class TestTrajectory:
    """Unit tests for Trajectory class"""
  
    def test_validate_valid_trajectory(self):
        traj = Trajectory(
            trajectory_id="test_1",
            asset_id="AAPL",
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
            label_h1=1,
            label_h5=-1,
            trend_regime=1,
            volatility_regime=0,
            structural_regime=-1
        )
        assert traj.validate() == True
  
    def test_validate_invalid_shape(self):
        traj = Trajectory(
            trajectory_id="test_1",
            asset_id="AAPL",
            matrix=np.random.randn(50, 45).astype(np.float32),  # Wrong shape
            embedding=np.random.randn(256).astype(np.float32),
            label_h1=1,
            label_h5=-1,
            trend_regime=1,
            volatility_regime=0,
            structural_regime=-1
        )
        assert traj.validate() == False
  
    def test_validate_invalid_regime(self):
        traj = Trajectory(
            trajectory_id="test_1",
            asset_id="AAPL",
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
            label_h1=1,
            label_h5=-1,
            trend_regime=5,  # Invalid regime
            volatility_regime=0,
            structural_regime=-1
        )
        assert traj.validate() == False


class TestSparseIndex:
    """Unit tests for SparseRegimeIndex"""
  
    @pytest.fixture
    def index(self):
        idx = SparseRegimeIndex()
        idx.add("t1", trend=1, volatility=0, structural=1, asset_id="AAPL", asset_class="equity")
        idx.add("t2", trend=1, volatility=1, structural=1, asset_id="AAPL", asset_class="equity")
        idx.add("t3", trend=-1, volatility=0, structural=0, asset_id="MSFT", asset_class="equity")
        idx.add("t4", trend=0, volatility=-1, structural=1, asset_id="BTC", asset_class="crypto")
        return idx
  
    def test_filter_exact_match(self, index):
        result = index.filter_by_regime(trend=1, volatility=0, structural=1, allow_adjacent=False)
        assert result == {"t1"}
  
    def test_filter_with_adjacent(self, index):
        result = index.filter_by_regime(trend=1, volatility=0, structural=1, allow_adjacent=True)
        assert "t1" in result
        assert "t2" in result  # Adjacent volatility
  
    def test_filter_by_asset(self, index):
        result = index.filter_by_asset(["AAPL"])
        assert result == {"t1", "t2"}
  
    def test_combined_filter(self, index):
        result = index.combined_filter(
            trend=1,
            asset_classes=["equity"],
            allow_adjacent=True
        )
        assert "t1" in result
        assert "t2" in result
        assert "t4" not in result  # crypto


class TestHybridDistance:
    """Unit tests for HybridDistanceCalculator"""
  
    @pytest.fixture
    def calculator(self):
        return HybridDistanceCalculator()
  
    def test_identical_matrices(self, calculator):
        matrix = np.random.randn(60, 45).astype(np.float32)
        regime = (1, 0, 1)
      
        dist, _ = calculator.compute(matrix, matrix, regime, regime)
        assert dist < 0.001  # Should be very close to 0
  
    def test_distance_symmetry(self, calculator):
        a = np.random.randn(60, 45).astype(np.float32)
        b = np.random.randn(60, 45).astype(np.float32)
        regime_a = (1, 0, 1)
        regime_b = (0, 1, 0)
      
        dist_ab, _ = calculator.compute(a, b, regime_a, regime_b)
        dist_ba, _ = calculator.compute(b, a, regime_b, regime_a)
      
        assert abs(dist_ab - dist_ba) < 0.001
  
    def test_regime_penalty(self, calculator):
        matrix = np.random.randn(60, 45).astype(np.float32)
        matrix2 = matrix + np.random.randn(60, 45).astype(np.float32) * 0.1
      
        # Same regime
        dist_same, _ = calculator.compute(matrix, matrix2, (1, 0, 1), (1, 0, 1))
      
        # Different regime
        dist_diff, _ = calculator.compute(matrix, matrix2, (1, 0, 1), (-1, -1, -1))
      
        assert dist_diff > dist_same  # Penalty should increase distance
```

### 10.2 Integration Tests

```python
class TestHPVDIntegration:
    """Integration tests for complete HPVD pipeline"""
  
    @pytest.fixture
    def hpvd_with_data(self):
        """Create HPVD with synthetic test data"""
        # Generate synthetic trajectories
        trajectories = []
        for i in range(1000):
            traj = Trajectory(
                trajectory_id=f"traj_{i}",
                asset_id=np.random.choice(["AAPL", "MSFT", "GOOGL"]),
                matrix=np.random.randn(60, 45).astype(np.float32),
                embedding=np.random.randn(256).astype(np.float32),
                label_h1=np.random.choice([-1, 1]),
                label_h5=np.random.choice([-1, 1]),
                return_h1=np.random.randn() * 0.02,
                return_h5=np.random.randn() * 0.05,
                trend_regime=np.random.choice([-1, 0, 1]),
                volatility_regime=np.random.choice([-1, 0, 1]),
                structural_regime=np.random.choice([-1, 0, 1]),
                asset_class="equity"
            )
            trajectories.append(traj)
      
        hpvd = HPVDEngine()
        hpvd.build(trajectories)
        return hpvd
  
    def test_search_returns_k_results(self, hpvd_with_data):
        query = Trajectory(
            trajectory_id="query",
            asset_id="TEST",
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
            trend_regime=1,
            volatility_regime=0,
            structural_regime=1
        )
      
        result = hpvd_with_data.search(query, k=25)
        assert result.k_returned == 25
      
        # Verify forecast results are included
        assert result.forecast_h1 is not None
        assert result.forecast_h5 is not None
        assert 0.0 <= result.forecast_h1.p_up <= 1.0
        assert 0.0 <= result.forecast_h5.p_up <= 1.0
      
        # Verify confidence intervals
        assert result.forecast_h1.confidence_interval[0] <= result.forecast_h1.p_up
        assert result.forecast_h1.p_up <= result.forecast_h1.confidence_interval[1]
      
        # Verify quality metrics
        assert 0.0 <= result.aci <= 1.0
        assert 0.0 <= result.regime_coherence <= 1.0
  
    def test_search_latency(self, hpvd_with_data):
        query = Trajectory(
            trajectory_id="query",
            asset_id="TEST",
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
            trend_regime=1,
            volatility_regime=0,
            structural_regime=1
        )
      
        result = hpvd_with_data.search(query, k=25)
        assert result.latency_ms < 100  # Should be under 100ms
  
    def test_save_and_load(self, hpvd_with_data, tmp_path):
        # Save
        save_path = str(tmp_path / "hpvd_test")
        hpvd_with_data.save(save_path)
      
        # Load
        hpvd_loaded = HPVDEngine()
        hpvd_loaded.load(save_path)
      
        # Verify
        assert len(hpvd_loaded.trajectories) == len(hpvd_with_data.trajectories)
        assert hpvd_loaded.dense_index.ntotal == hpvd_with_data.dense_index.ntotal
```

---

## 11. Validation Checklist

Based on source requirements, HPVD must pass these validation criteria:

### 11.1 HPVD Correctness

| Metric              | Target                    | Source      |
| ------------------- | ------------------------- | ----------- |
| Trajectory density  | > 95% (no major gaps)     | Sprint Plan |
| Embedding variance  | > 10^-5 per dimension     | Sprint Plan |
| Query latency       | < 50ms @ 2M+ trajectories | Sprint Plan |
| ACI                 | > 0.7 for 80%+ queries    | Sprint Plan |
| Regime coherence RC | > 0.65                    | Sprint Plan |
| ECE                 | < 8% on validation set    | Sprint Plan |

### 11.2 Integration Correctness

| Metric               | Target                      | Source      |
| -------------------- | --------------------------- | ----------- |
| End-to-end latency   | < 200ms                     | Sprint Plan |
| Determinism          | Same input → same output   | Sprint Plan |
| Graceful degradation | Handle < K analogues        | Sprint Plan |
| Input validation     | Return 4xx on invalid input | Sprint Plan |

### 11.3 Financial Performance

| Metric                 | Target                     | Source       |
| ---------------------- | -------------------------- | ------------ |
| H1 Accuracy            | > 52% (vs 50% random walk) | Matrix22 MVP |
| H5 Accuracy            | > 54%                      | Matrix22 MVP |
| Brier Score            | < 0.18                     | Sprint Plan  |
| Cross-regime stability | < 5-7pp drop               | Matrix22 MVP |
| Crisis performance     | ECE < 7%                   | Sprint Plan  |

### 11.4 PMR-DB Integration (Future)

| Metric             | Target                  | Source      |
| ------------------ | ----------------------- | ----------- |
| P(H1) + P(H1_down) | ≈ 1.0                  | Sprint Plan |
| ECE                | < 5% across all regimes | Sprint Plan |
| Brier Score        | < 0.18                  | Sprint Plan |
| Abstention trigger | entropy > 0.9           | Sprint Plan |
| Precision          | > 95% at 80% coverage   | Sprint Plan |

---

## Appendix: Quick Reference

### A. Key Formulas

```
HYBRID DISTANCE
===============
d_total = (0.3×d̂_euc + 0.4×d̂_cos + 0.3×d̂_temp) × (1 + 0.2×(1 - regime_match))

where:
  d̂_euc = ||vec(A) - vec(B)||₂ / (√2700 × 2)
  d̂_cos = (1 - cos(vec(A), vec(B))) / 2
  d̂_temp = Σᵢ wᵢ ||Aᵢ - Bᵢ||₂ / (√45 × 2)
  wᵢ = 0.95^(59-i) / Σⱼ 0.95^(59-j)


ANALOG COHESION INDEX (ACI)
===========================
ACI = 1 - (mean(distances) + std(distances)) / 2
Target: ACI > 0.7 for 80% of queries


REGIME MATCH SCORE
==================
score = mean([1 - |Rₐᵢ - Rᵦᵢ|/2 for i in 0,1,2])
Target: RC > 0.65


BRIER SCORE (Calibration)
=========================
BS = (1/N) × Σᵢ(pᵢ - yᵢ)²
Target: BS < 0.18 (competitive with LSTM)


EXPECTED CALIBRATION ERROR (ECE)
================================
ECE = Σₘ (|Bₘ|/N) × |accuracy(Bₘ) - confidence(Bₘ)|
Target: ECE < 8% (HPVD), ECE < 5% (full system)


WEIGHTED PROBABILITY (Forecast)
===============================
P(up) = Σᵢ wᵢ × outcomeᵢ
where wᵢ = exp(-α × distanceᵢ) / Σⱼ exp(-α × distanceⱼ)
α = 2.0 (decay factor)


ABSTENTION RULE
===============
If entropy > 0.9 → abstain (return "LOW_CONFIDENCE")
entropy = -p×log₂(p) - (1-p)×log₂(1-p)
```

### B. Default Parameters

| Parameter                | Default | Description              | Source                    |
| ------------------------ | ------- | ------------------------ | ------------------------- |
| `default_k`            | 25      | Number of neighbors      | Sprint Plan               |
| `search_k_multiplier`  | 3       | Oversample factor        |                           |
| `weight_euclidean`     | 0.3     | Euclidean weight         |                           |
| `weight_cosine`        | 0.4     | Cosine weight            |                           |
| `weight_temporal`      | 0.3     | Temporal weight          |                           |
| `regime_penalty`       | 0.2     | Regime mismatch penalty  |                           |
| `temporal_decay`       | 0.95    | Temporal weight decay    |                           |
| `min_aci`              | 0.7     | Minimum ACI threshold    | ACI > 0.7 for 80% queries |
| `min_regime_coherence` | 0.65    | Minimum regime coherence | RC > 0.65                 |
| `max_ece`              | 0.08    | Maximum ECE              | ECE < 8%                  |
| `max_brier`            | 0.18    | Maximum Brier Score      | Brier < 0.18              |
| `abstention_entropy`   | 0.9     | Abstention threshold     | entropy > 0.9 → abstain  |

### C. Complexity Summary

| Operation     | Time                | Space |
| ------------- | ------------------- | ----- |
| Sparse filter | O(27)               | O(K)  |
| Dense search  | O(N×D) or O(log N) | O(K)  |
| Hybrid rerank | O(K×D²)           | O(K)  |
| Total search  | O(N×D)             | O(K)  |

---

**End of Document**

*Version 1.0.0-MVP | December 2024 | Matrix22*
