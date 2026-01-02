# HPVD Architecture Document

## Hybrid Probabilistic Vector Database for Trajectory Intelligence

**Version:** 1.0.0-MVP
**Date:** December 2025
**Project:** Matrix22
**Related:** [HPVD_Technical_Specification.md](./HPVD_Technical_Specification.md) - Detailed implementation specs

**Source Documents:**

- [KALIBRY FINANCIAL MVP](../sources/KALIBRY%20FINANCIAL%20MVP.docx-20251209213923.md)
- [HPVD + PMR-DB Sprint Plan](../sources/How%20HPVD%20+%20PMR-DB%20Power%20Kalibry%20Finance-20251209213850.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Core Concepts](#3-core-concepts)
4. [Data Schema Design](#4-data-schema-design)
5. [Index Structures](#5-index-structures)
6. [HPVD Engine Design](#6-hpvd-engine-design)
7. [PMR-DB Design](#7-pmr-db-design)
8. [API Specification](#8-api-specification)
9. [Performance Requirements](#9-performance-requirements)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Executive Summary

### 1.1 Purpose

HPVD (Hybrid Probabilistic Vector Database) adalah komponen inti dari Matrix22 yang bertanggung jawab untuk:

1. **Menyimpan** jutaan trajectory historis dari berbagai aset finansial
2. **Mencari** trajectory yang paling mirip dengan kondisi pasar saat ini
3. **Menghasilkan** probabilitas prediksi yang terkalibrasi berdasarkan outcome historis

### 1.2 Key Specifications (MVP)

| Parameter            | MVP Target                | Production Target |
| -------------------- | ------------------------- | ----------------- |
| Trajectory Dimension | 60 × 45 (2,700 features) | Same              |
| Reduced Dimension    | 256                       | 128-256           |
| Query Latency        | < 50ms                    | < 20ms            |
| Database Scale       | 100K trajectories         | 10M+ trajectories |
| Recall@K             | > 85%                     | > 90%             |
| K (neighbors)        | 25                        | 25-40             |

### 1.3 MVP Scope

**Included:**

- Basic trajectory storage and retrieval
- Single FAISS index (no sharding)
- Regime-based filtering (sparse index)
- Hybrid distance computation
- Basic probability calibration (Isotonic Regression)
- Simple API endpoints

**Not Included (Future):**

- Multi-shard distributed storage
- Advanced compression (PQ/OPQ)
- Cross-encoder reranking
- Real-time streaming updates
- Multi-modal reasoning (text + events)

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  REST API   │  │  Dashboard  │  │  Backtest   │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API GATEWAY                                │
│                    POST /v1/forecast                             │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MATRIX22 ENGINE                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   EMBEDDING ENGINE                        │   │
│  │  OHLCV → R45 Features → Trajectory (60×45) → Embed (256) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        HPVD                               │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │   │
│  │  │   Sparse    │──▶│    Dense    │──▶│   Hybrid    │     │   │
│  │  │   Filter    │   │  Retrieval  │   │  Reranking  │     │   │
│  │  └─────────────┘   └─────────────┘   └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                       PMR-DB                              │   │
│  │        Aggregate Outcomes → Calibrate → Uncertainty       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │  PostgreSQL │   │    FAISS    │   │  File Store │            │
│  │  (metadata) │   │   (vectors) │   │  (matrices) │            │
│  └─────────────┘   └─────────────┘   └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component                  | Responsibility                                             |
| -------------------------- | ---------------------------------------------------------- |
| **Embedding Engine** | Transform OHLCV data into R45 features and trajectories    |
| **HPVD**             | Find similar trajectories using hybrid sparse+dense search |
| **PMR-DB**           | Aggregate outcomes and produce calibrated probabilities    |
| **Storage Layer**    | Persist trajectories, vectors, and metadata                |

### 2.3 Data Flow

```
1. INPUT: Query Trajectory T_q (60×45 matrix)
      │
      ▼
2. EMBED: Reduce to 256-dim vector
      │
      ▼
3. SPARSE FILTER: Filter by regime (trend, volatility, structure)
      │
      ▼
4. DENSE SEARCH: FAISS k-NN search → Top 75 candidates
      │
      ▼
5. RERANK: Hybrid distance computation → Top 25 analogs
      │
      ▼
6. AGGREGATE: Weighted outcome aggregation
      │
      ▼
7. CALIBRATE: Isotonic regression calibration
      │
      ▼
8. OUTPUT: {p_h1, p_h5, uncertainty, analogs, regimes}
```

---

## 3. Core Concepts

### 3.1 Trajectory Definition

Trajectory adalah unit fundamental dalam Trajectory Intelligence:

```
T ∈ ℝ^(60×45)
```

- **60 rows**: 60 hari trading berurutan
- **45 columns**: 45 engineered features (R45 embedding)

### 3.2 R45 Feature Blocks

| Block           | Features        | Count        | Description                            |
| --------------- | --------------- | ------------ | -------------------------------------- |
| A               | Returns         | 8            | 1d, 5d, 10d, 20d returns (plain & log) |
| B               | Trend           | 10           | Slopes, R², MA crossovers             |
| C               | Volatility      | 12           | Realized vol, ATR, shocks, gaps        |
| D               | Price Structure | 10           | Candle patterns, skew, kurtosis        |
| E               | Regime          | 5            | Trend/vol/momentum/structure regimes   |
| **Total** |                 | **45** |                                        |

### 3.3 Labels (Outcomes)

Setiap trajectory memiliki label berdasarkan pergerakan harga setelah window:

- **H1 (1-day)**: `label_h1 = sign(close_{t+1} - close_t)`
- **H5 (5-day)**: `label_h5 = sign(close_{t+5} - close_t)`

### 3.4 Regimes

Tiga tipe regime untuk filtering:

| Regime     | Values                                  | Based On        |
| ---------- | --------------------------------------- | --------------- |
| Trend      | UP (+1), SIDEWAYS (0), DOWN (-1)        | 60d slope       |
| Volatility | HIGH (+1), MEDIUM (0), LOW (-1)         | Percentile vol  |
| Structural | TREND (+1), MIXED (0), MEAN_REVERT (-1) | Autocorrelation |

---

## 4. Data Schema Design

### 4.1 Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        TRAJECTORY                           │
├─────────────────────────────────────────────────────────────┤
│ trajectory_id    : UUID (PK)                                │
│ asset_id         : VARCHAR(32)                              │
│ end_timestamp    : TIMESTAMP                                │
│ matrix_path      : VARCHAR(512)    -- Path to .npy file     │
│ embedding_256    : FLOAT[256]      -- Reduced embedding     │
│ label_h1         : SMALLINT        -- +1 or -1              │
│ label_h5         : SMALLINT        -- +1 or -1              │
│ return_h1        : FLOAT           -- Actual return         │
│ return_h5        : FLOAT           -- Actual return         │
│ trend_regime     : SMALLINT        -- -1, 0, +1             │
│ volatility_regime: SMALLINT        -- -1, 0, +1             │
│ structural_regime: SMALLINT        -- -1, 0, +1             │
│ asset_class      : VARCHAR(16)     -- equity/forex/crypto   │
│ created_at       : TIMESTAMP                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FORECAST_LOG                           │
├─────────────────────────────────────────────────────────────┤
│ log_id           : UUID (PK)                                │
│ query_asset_id   : VARCHAR(32)                              │
│ query_timestamp  : TIMESTAMP                                │
│ p_up_h1          : FLOAT                                    │
│ p_up_h5          : FLOAT                                    │
│ aci              : FLOAT           -- Analog Cohesion Index │
│ k_used           : SMALLINT                                 │
│ latency_ms       : FLOAT                                    │
│ created_at       : TIMESTAMP                                │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Storage Strategy (MVP)

```
matrix22-data/
├── trajectories/
│   ├── matrices/           # .npy files (60×45 matrices)
│   │   ├── AAPL/
│   │   │   ├── 2020-01-15.npy
│   │   │   ├── 2020-01-16.npy
│   │   │   └── ...
│   │   ├── BTC-USD/
│   │   └── ...
│   │
│   └── embeddings/         # Precomputed 256-dim embeddings
│       └── all_embeddings.npy    # (N, 256) matrix
│
├── indexes/
│   ├── faiss_index.bin     # FAISS index
│   └── id_mapping.pkl      # FAISS ID → trajectory_id
│
├── calibration/
│   ├── isotonic_h1.pkl     # Calibrator for H1
│   └── isotonic_h5.pkl     # Calibrator for H5
│
└── metadata/
    └── trajectories.db     # SQLite for MVP (PostgreSQL for prod)
```

### 4.3 Python Data Classes

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import numpy as np

@dataclass
class Trajectory:
    """Core trajectory entity"""
    trajectory_id: str
    asset_id: str
    end_timestamp: datetime
  
    # Data
    matrix: np.ndarray              # Shape: (60, 45)
    embedding: np.ndarray           # Shape: (256,)
  
    # Labels
    label_h1: int                   # +1 or -1
    label_h5: int                   # +1 or -1
    return_h1: float
    return_h5: float
  
    # Regimes
    trend_regime: int               # -1, 0, +1
    volatility_regime: int          # -1, 0, +1
    structural_regime: int          # -1, 0, +1
  
    # Metadata
    asset_class: str                # "equity", "forex", "crypto", "index"

@dataclass
class AnalogResult:
    """Single analog from similarity search"""
    trajectory_id: str
    asset_id: str
    distance: float
    label_h1: int
    label_h5: int
    return_h1: float
    return_h5: float
    regime_match: float             # 0-1 score

@dataclass
class ForecastResult:
    """Complete forecast output"""
    # Probabilities
    p_up_h1: float
    p_up_h5: float
  
    # Uncertainty
    entropy_h1: float
    entropy_h5: float
    aci: float                      # Analog Cohesion Index
  
    # Evidence
    analogs: List[AnalogResult]
    k_used: int
  
    # Regimes
    trend_regime: str               # "UP", "SIDEWAYS", "DOWN"
    volatility_regime: str          # "HIGH", "MEDIUM", "LOW"
  
    # Meta
    latency_ms: float
```

---

## 5. Index Structures

### 5.1 Dense Index (FAISS)

#### Configuration (MVP)

```python
import faiss

# MVP: Simple flat index (exact search)
# Suitable for < 500K trajectories
dimension = 256
index = faiss.IndexFlatIP(dimension)  # Inner Product after L2 normalization

# Add vectors
embeddings = np.load("all_embeddings.npy")  # (N, 256)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Search
query = np.array([...])  # (256,)
faiss.normalize_L2(query.reshape(1, -1))
distances, indices = index.search(query.reshape(1, -1), k=75)
```

#### Index Type Selection

| Scale     | Index Type     | Latency | Recall |
| --------- | -------------- | ------- | ------ |
| < 100K    | IndexFlatIP    | < 5ms   | 100%   |
| 100K - 1M | IndexIVFFlat   | < 10ms  | ~95%   |
| 1M - 10M  | IndexHNSW      | < 15ms  | ~92%   |
| > 10M     | IndexHNSW + PQ | < 20ms  | ~90%   |

**MVP Choice:** `IndexFlatIP` (exact search, simplest implementation)

### 5.2 Sparse Index (Regime Filter)

#### In-Memory Inverted Index

```python
from collections import defaultdict
from typing import Set, Dict, Tuple

class SparseRegimeIndex:
    """Simple inverted index for regime-based filtering"""
  
    def __init__(self):
        # Key: (trend, vol, struct) tuple
        # Value: Set of trajectory_ids
        self.regime_index: Dict[Tuple[int, int, int], Set[str]] = defaultdict(set)
      
        # Asset-based index
        self.asset_index: Dict[str, Set[str]] = defaultdict(set)
      
        # Reverse lookup
        self.trajectory_regimes: Dict[str, Tuple[int, int, int]] = {}
  
    def add(self, trajectory_id: str, trend: int, vol: int, struct: int, asset_id: str):
        """Add trajectory to index"""
        key = (trend, vol, struct)
        self.regime_index[key].add(trajectory_id)
        self.asset_index[asset_id].add(trajectory_id)
        self.trajectory_regimes[trajectory_id] = key
  
    def filter(self, 
               trend: int = None, 
               vol: int = None, 
               struct: int = None,
               allow_adjacent: bool = True) -> Set[str]:
        """
        Filter trajectories by regime
      
        Args:
            trend: Target trend regime (-1, 0, +1) or None for any
            vol: Target volatility regime or None
            struct: Target structural regime or None
            allow_adjacent: Include adjacent regimes (±1)
      
        Returns:
            Set of matching trajectory IDs
        """
        result = set()
      
        for key, trajectories in self.regime_index.items():
            k_trend, k_vol, k_struct = key
            match = True
          
            if trend is not None:
                if allow_adjacent:
                    match = match and abs(k_trend - trend) <= 1
                else:
                    match = match and k_trend == trend
          
            if vol is not None:
                if allow_adjacent:
                    match = match and abs(k_vol - vol) <= 1
                else:
                    match = match and k_vol == vol
          
            if struct is not None:
                if allow_adjacent:
                    match = match and abs(k_struct - struct) <= 1
                else:
                    match = match and k_struct == struct
          
            if match:
                result.update(trajectories)
      
        return result
```

### 5.3 Index Interaction

```
Query Trajectory
      │
      ▼
┌─────────────────────────────────────┐
│     SPARSE INDEX (Regime Filter)    │
│                                     │
│  Query regime: (trend=1, vol=0)     │
│  Filter: allow_adjacent=True        │
│  Result: 50,000 trajectory IDs      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│     DENSE INDEX (FAISS)             │
│                                     │
│  Search in filtered subset          │
│  k = 75 candidates                  │
│  Result: [(id, distance), ...]      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│     HYBRID RERANKING                │
│                                     │
│  Compute hybrid distance            │
│  Apply regime penalty               │
│  Return top 25 analogs              │
└─────────────────────────────────────┘
```

---

## 6. HPVD Engine Design

### 6.1 Main Class Structure

```python
class HPVDEngine:
    """
    Hybrid Probabilistic Vector Database Engine
  
    Main entry point for trajectory similarity search
    """
  
    def __init__(self, config: HPVDConfig):
        self.config = config
      
        # Indexes
        self.dense_index: faiss.Index = None
        self.sparse_index: SparseRegimeIndex = None
      
        # Mappings
        self.id_to_idx: Dict[str, int] = {}      # trajectory_id → faiss index
        self.idx_to_id: Dict[int, str] = {}      # faiss index → trajectory_id
        self.trajectories: Dict[str, Trajectory] = {}  # Full trajectory data
      
        # Distance calculator
        self.distance_calc = HybridDistanceCalculator(config.distance_config)
  
    def build_index(self, trajectories: List[Trajectory]):
        """Build FAISS and sparse indexes from trajectories"""
        pass
  
    def search(self, 
               query_trajectory: Trajectory,
               k: int = 25) -> List[AnalogResult]:
        """
        Find k most similar trajectories
      
        Pipeline:
        1. Sparse filter by regime
        2. Dense FAISS search
        3. Hybrid reranking
        4. Quality filtering
        """
        pass
  
    def save(self, path: str):
        """Save indexes to disk"""
        pass
  
    def load(self, path: str):
        """Load indexes from disk"""
        pass
```

### 6.2 Search Pipeline Implementation

```python
def search(self, 
           query_trajectory: Trajectory,
           k: int = 25) -> List[AnalogResult]:
    """Find k most similar trajectories"""
  
    # ========== STAGE 1: Sparse Filtering ==========
    candidate_ids = self.sparse_index.filter(
        trend=query_trajectory.trend_regime,
        vol=query_trajectory.volatility_regime,
        struct=query_trajectory.structural_regime,
        allow_adjacent=True
    )
  
    # If too few candidates, relax filtering
    if len(candidate_ids) < k * 10:
        candidate_ids = self.sparse_index.filter(
            trend=query_trajectory.trend_regime,
            vol=None,  # Relax volatility
            struct=None,  # Relax structure
            allow_adjacent=True
        )
  
    # ========== STAGE 2: Dense Retrieval ==========
    # Get FAISS indices for candidates
    candidate_indices = [self.id_to_idx[tid] for tid in candidate_ids 
                         if tid in self.id_to_idx]
  
    # Create selector for subset search
    query_embedding = query_trajectory.embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)
  
    # Search (get more than k for reranking)
    search_k = min(k * 3, len(candidate_indices))
    distances, indices = self.dense_index.search(query_embedding, search_k)
  
    # ========== STAGE 3: Hybrid Reranking ==========
    candidates = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
      
        tid = self.idx_to_id.get(idx)
        if tid is None:
            continue
      
        traj = self.trajectories[tid]
      
        # Compute hybrid distance
        hybrid_dist, components = self.distance_calc.compute(
            query_trajectory.matrix,
            traj.matrix,
            query_trajectory.get_regime_tuple(),
            traj.get_regime_tuple()
        )
      
        candidates.append({
            'trajectory_id': tid,
            'asset_id': traj.asset_id,
            'distance': hybrid_dist,
            'faiss_distance': float(dist),
            'label_h1': traj.label_h1,
            'label_h5': traj.label_h5,
            'return_h1': traj.return_h1,
            'return_h5': traj.return_h5,
            'regime_match': components['regime_match']
        })
  
    # Sort by hybrid distance
    candidates.sort(key=lambda x: x['distance'])
  
    # ========== STAGE 4: Return Top K ==========
    results = [
        AnalogResult(
            trajectory_id=c['trajectory_id'],
            asset_id=c['asset_id'],
            distance=c['distance'],
            label_h1=c['label_h1'],
            label_h5=c['label_h5'],
            return_h1=c['return_h1'],
            return_h5=c['return_h5'],
            regime_match=c['regime_match']
        )
        for c in candidates[:k]
    ]
  
    return results
```

### 6.3 Hybrid Distance Calculation

```python
class HybridDistanceCalculator:
    """Compute hybrid distance between trajectories"""
  
    def __init__(self, config: DistanceConfig = None):
        self.config = config or DistanceConfig()
      
        # Precompute temporal weights
        self.temporal_weights = self._compute_temporal_weights(60)
  
    def _compute_temporal_weights(self, window: int) -> np.ndarray:
        """Recent days weighted more heavily"""
        decay = 0.95
        weights = np.array([decay ** (window - 1 - t) for t in range(window)])
        return weights / weights.sum()
  
    def compute(self,
                matrix_a: np.ndarray,  # (60, 45)
                matrix_b: np.ndarray,  # (60, 45)
                regime_a: Tuple[int, int, int],
                regime_b: Tuple[int, int, int]) -> Tuple[float, dict]:
        """
        Compute hybrid distance
      
        Returns:
            (total_distance, component_dict)
        """
        # Flatten
        flat_a = matrix_a.flatten()
        flat_b = matrix_b.flatten()
      
        # Component distances
        d_euclidean = np.linalg.norm(flat_a - flat_b)
      
        d_cosine = 1.0 - np.dot(flat_a, flat_b) / (
            np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-9
        )
      
        # Time-weighted distance
        day_dists = np.linalg.norm(matrix_a - matrix_b, axis=1)
        d_temporal = np.dot(self.temporal_weights, day_dists)
      
        # Regime match score
        regime_match = self._regime_match_score(regime_a, regime_b)
      
        # Normalize and combine
        d_euc_norm = d_euclidean / (np.sqrt(2700) * 2)
        d_cos_norm = d_cosine / 2.0
        d_temp_norm = d_temporal / (np.sqrt(45) * 2)
      
        base = (
            self.config.weight_euclidean * d_euc_norm +
            self.config.weight_cosine * d_cos_norm +
            self.config.weight_temporal * d_temp_norm
        )
      
        # Apply regime penalty
        penalty = (1.0 - regime_match) * self.config.regime_penalty
        total = base * (1.0 + penalty)
      
        return total, {
            'euclidean': d_euclidean,
            'cosine': d_cosine,
            'temporal': d_temporal,
            'regime_match': regime_match,
            'base_distance': base,
            'total_distance': total
        }
  
    def _regime_match_score(self, a: Tuple, b: Tuple) -> float:
        """Compute regime match score (0-1)"""
        scores = []
        for va, vb in zip(a, b):
            scores.append(1.0 - abs(va - vb) / 2.0)
        return np.mean(scores)
```

---

## 7. PMR-DB Design

### 7.1 Probabilistic Aggregation

```python
class PMREngine:
    """
    Probabilistic Multimodal Reasoning Engine
  
    Aggregates analog outcomes into calibrated probabilities
    """
  
    def __init__(self, calibrator_h1=None, calibrator_h5=None):
        self.calibrator_h1 = calibrator_h1
        self.calibrator_h5 = calibrator_h5
  
    def aggregate(self, analogs: List[AnalogResult]) -> ForecastResult:
        """
        Aggregate analog outcomes into probabilistic forecast
        """
        if not analogs:
            return self._empty_result()
      
        # ========== Weighted Aggregation ==========
        weights = self._compute_weights(analogs)
      
        # H1 probability
        h1_outcomes = np.array([1 if a.label_h1 == 1 else 0 for a in analogs])
        p_raw_h1 = np.average(h1_outcomes, weights=weights)
      
        # H5 probability
        h5_outcomes = np.array([1 if a.label_h5 == 1 else 0 for a in analogs])
        p_raw_h5 = np.average(h5_outcomes, weights=weights)
      
        # ========== Calibration ==========
        p_cal_h1 = self._calibrate(p_raw_h1, self.calibrator_h1)
        p_cal_h5 = self._calibrate(p_raw_h5, self.calibrator_h5)
      
        # ========== Uncertainty ==========
        entropy_h1 = self._entropy(p_cal_h1)
        entropy_h5 = self._entropy(p_cal_h5)
      
        distances = np.array([a.distance for a in analogs])
        aci = self._compute_aci(distances)
      
        # ========== Determine Regimes ==========
        trend = self._majority_vote([a.regime_match for a in analogs], 'trend')
        vol = self._majority_vote([a.regime_match for a in analogs], 'vol')
      
        return ForecastResult(
            p_up_h1=p_cal_h1,
            p_up_h5=p_cal_h5,
            entropy_h1=entropy_h1,
            entropy_h5=entropy_h5,
            aci=aci,
            analogs=analogs,
            k_used=len(analogs),
            trend_regime=trend,
            volatility_regime=vol,
            latency_ms=0.0  # Set by caller
        )
  
    def _compute_weights(self, analogs: List[AnalogResult]) -> np.ndarray:
        """Inverse distance weighting"""
        distances = np.array([a.distance for a in analogs])
      
        # Exponential decay
        alpha = 5.0
        weights = np.exp(-alpha * distances)
      
        # Normalize
        return weights / weights.sum()
  
    def _calibrate(self, p_raw: float, calibrator) -> float:
        """Apply isotonic calibration"""
        if calibrator is None:
            return p_raw
      
        try:
            return float(calibrator.predict([[p_raw]])[0])
        except:
            return p_raw
  
    def _entropy(self, p: float) -> float:
        """Binary entropy"""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
  
    def _compute_aci(self, distances: np.ndarray) -> float:
        """
        Analog Cohesion Index
      
        High ACI = analogs are similar to each other = more reliable
        """
        if len(distances) < 2:
            return 1.0
      
        mean_dist = distances.mean()
        std_dist = distances.std()
      
        # Normalize (assume max distance ~2)
        cohesion = 1.0 - (mean_dist + std_dist) / 2.0
        return float(np.clip(cohesion, 0.0, 1.0))
```

### 7.2 Calibration (Isotonic Regression)

```python
from sklearn.isotonic import IsotonicRegression

class CalibrationManager:
    """Manage probability calibration models"""
  
    def __init__(self):
        self.calibrator_h1 = None
        self.calibrator_h5 = None
  
    def fit(self, 
            raw_probs_h1: np.ndarray,
            true_labels_h1: np.ndarray,
            raw_probs_h5: np.ndarray,
            true_labels_h5: np.ndarray):
        """
        Fit calibrators on validation data
      
        Args:
            raw_probs_h1: Uncalibrated probabilities for H1
            true_labels_h1: Actual outcomes (0 or 1)
            ...
        """
        # H1 calibrator
        self.calibrator_h1 = IsotonicRegression(
            y_min=0.0, 
            y_max=1.0, 
            out_of_bounds='clip'
        )
        self.calibrator_h1.fit(raw_probs_h1, true_labels_h1)
      
        # H5 calibrator
        self.calibrator_h5 = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )
        self.calibrator_h5.fit(raw_probs_h5, true_labels_h5)
  
    def save(self, path: str):
        """Save calibrators to disk"""
        import pickle
        with open(f"{path}/isotonic_h1.pkl", 'wb') as f:
            pickle.dump(self.calibrator_h1, f)
        with open(f"{path}/isotonic_h5.pkl", 'wb') as f:
            pickle.dump(self.calibrator_h5, f)
  
    def load(self, path: str):
        """Load calibrators from disk"""
        import pickle
        with open(f"{path}/isotonic_h1.pkl", 'rb') as f:
            self.calibrator_h1 = pickle.load(f)
        with open(f"{path}/isotonic_h5.pkl", 'rb') as f:
            self.calibrator_h5 = pickle.load(f)
```

---

## 8. API Specification

### 8.1 Endpoints (MVP)

| Method | Endpoint         | Description                        |
| ------ | ---------------- | ---------------------------------- |
| POST   | `/v1/forecast` | Generate forecast for a trajectory |
| GET    | `/v1/health`   | Health check                       |
| GET    | `/v1/stats`    | System statistics                  |

### 8.2 Forecast Endpoint

**Request:**

```json
POST /v1/forecast
Content-Type: application/json

{
  "mode": "TRAJECTORY",
  "asset_id": "AAPL",
  "trajectory": [[...45 values...], [...], ...],  // 60×45 matrix
  "options": {
    "k": 25,
    "include_analogs": true,
    "include_uncertainty": true
  }
}
```

**Response:**

```json
{
  "status": "SUCCESS",
  "data": {
    "probabilities": {
      "h1": {
        "p_up": 0.62,
        "p_down": 0.38
      },
      "h5": {
        "p_up": 0.68,
        "p_down": 0.32
      }
    },
    "uncertainty": {
      "entropy_h1": 0.88,
      "entropy_h5": 0.76,
      "aci": 0.71
    },
    "regimes": {
      "trend": "UP",
      "volatility": "MEDIUM",
      "structure": "TREND_FOLLOWING"
    },
    "analogs": [
      {
        "trajectory_id": "abc123",
        "asset_id": "AAPL",
        "timestamp": "2020-03-15",
        "distance": 0.12,
        "outcome_h1": 1,
        "outcome_h5": 1
      }
    ],
    "k_used": 25
  },
  "meta": {
    "latency_ms": 35.2,
    "model_version": "1.0.0-mvp",
    "request_id": "req_123abc"
  }
}
```

### 8.3 Error Responses

```json
{
  "status": "ERROR",
  "error": {
    "code": "INVALID_TRAJECTORY",
    "message": "Trajectory matrix must be 60×45",
    "details": {
      "received_shape": [55, 45]
    }
  }
}
```

---

## 9. Performance Requirements

### 9.1 Latency Budget (MVP)

| Component           | Target         | Maximum        |
| ------------------- | -------------- | -------------- |
| Input validation    | 1ms            | 2ms            |
| Sparse filtering    | 2ms            | 5ms            |
| Dense retrieval     | 15ms           | 25ms           |
| Hybrid reranking    | 5ms            | 10ms           |
| PMR aggregation     | 2ms            | 5ms            |
| Response formatting | 1ms            | 2ms            |
| **Total**     | **26ms** | **49ms** |

### 9.2 Scale Requirements (MVP)

| Metric             | MVP Target |
| ------------------ | ---------- |
| Trajectories       | 100K       |
| Assets             | 50+        |
| Concurrent queries | 10         |
| Memory (index)     | < 8GB      |
| Disk (data)        | < 50GB     |

### 9.3 Quality Metrics (MVP)

| Metric        | Target |
| ------------- | ------ |
| Recall@25     | > 85%  |
| Latency P95   | < 50ms |
| ACI (average) | > 0.65 |

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Setup project structure
- [ ] Implement data classes (Trajectory, AnalogResult, ForecastResult)
- [ ] Create SQLite storage for metadata
- [ ] Implement trajectory file I/O (.npy)

### Phase 2: Indexes (Week 3-4)

- [ ] Implement SparseRegimeIndex
- [ ] Integrate FAISS IndexFlatIP
- [ ] Build index from sample data
- [ ] Test search functionality

### Phase 3: HPVD Engine (Week 5-6)

- [ ] Implement HPVDEngine class
- [ ] Implement HybridDistanceCalculator
- [ ] Implement search pipeline
- [ ] Unit tests for search accuracy

### Phase 4: PMR-DB (Week 7-8)

- [ ] Implement PMREngine
- [ ] Implement CalibrationManager
- [ ] Train calibrators on validation data
- [ ] Test probability calibration

### Phase 5: API & Integration (Week 9-10)

- [ ] Implement FastAPI endpoints
- [ ] End-to-end integration tests
- [ ] Performance benchmarks
- [ ] Documentation

### Phase 6: Validation (Week 11-12)

- [ ] Accuracy benchmarks vs baselines
- [ ] Calibration quality tests
- [ ] Latency optimization
- [ ] Bug fixes and polish

---

## Appendix A: Configuration Reference

```python
from dataclasses import dataclass

@dataclass
class HPVDConfig:
    """HPVD configuration"""
  
    # Dimensions
    trajectory_window: int = 60
    feature_count: int = 45
    embedding_dim: int = 256
  
    # Search
    default_k: int = 25
    search_k_multiplier: int = 3
  
    # Distance weights
    weight_euclidean: float = 0.3
    weight_cosine: float = 0.4
    weight_temporal: float = 0.3
    regime_penalty: float = 0.2
  
    # Filtering
    allow_adjacent_regimes: bool = True
    min_candidates: int = 100
  
    # Paths
    index_path: str = "data/indexes"
    data_path: str = "data/trajectories"
  
    # Performance
    faiss_nprobe: int = 16  # For IVF indexes
    batch_size: int = 32

@dataclass 
class DistanceConfig:
    """Distance calculation configuration"""
    weight_euclidean: float = 0.3
    weight_cosine: float = 0.4
    weight_temporal: float = 0.3
    regime_penalty: float = 0.2
    temporal_decay: float = 0.95
```

---

## Appendix B: File Formats

### Trajectory Matrix (.npy)

```python
# Save
matrix = np.zeros((60, 45), dtype=np.float32)
np.save("AAPL/2020-01-15.npy", matrix)

# Load
matrix = np.load("AAPL/2020-01-15.npy")
```

### Embedding File (.npy)

```python
# All embeddings in single file
embeddings = np.zeros((100000, 256), dtype=np.float32)
np.save("all_embeddings.npy", embeddings)
```

### ID Mapping (.pkl)

```python
import pickle

mapping = {
    0: "traj_abc123",
    1: "traj_def456",
    # ...
}

with open("id_mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)
```

---

## Appendix C: Quality Gates

### Analog Quality Checks

```python
def validate_analogs(analogs: List[AnalogResult], config: HPVDConfig) -> bool:
    """
    Validate analog quality before aggregation
  
    Returns True if analogs pass quality gates
    """
    if len(analogs) < config.min_k:
        return False  # Not enough analogs
  
    # Check ACI
    distances = np.array([a.distance for a in analogs])
    aci = 1.0 - (distances.mean() + distances.std()) / 2.0
    if aci < config.min_aci:
        return False  # Analogs too dispersed
  
    # Check regime coherence
    regime_scores = np.array([a.regime_match for a in analogs])
    if regime_scores.mean() < config.min_regime_coherence:
        return False  # Regimes don't match well
  
    return True
```

---

**End of Document**

*Version 1.0.0-MVP | December 2024 | Matrix22*
