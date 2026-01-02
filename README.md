# HPVD-M22

**Hybrid Probabilistic Vector Database for Trajectory Intelligence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

HPVD (Hybrid Probabilistic Vector Database) is a high-performance vector search engine designed for finding historical analogs in financial trajectory data. It combines:

- **Sparse Filtering**: Regime-based inverted index for fast candidate filtering
- **Dense Retrieval**: FAISS-based approximate nearest neighbor search
- **Hybrid Reranking**: Multi-component distance metrics for precise similarity

## Project Structure

```
HPVD-M22/
├── src/
│   ├── hpvd/                    # Core HPVD library
│   │   ├── __init__.py
│   │   ├── trajectory.py        # Trajectory data model
│   │   ├── sparse_index.py      # Regime-based inverted index
│   │   ├── dense_index.py       # FAISS wrapper
│   │   ├── distance.py          # Hybrid distance calculator
│   │   └── engine.py            # Main HPVD engine
│   └── prototypes/
│       └── bm25_prototype.py    # BM25 text retrieval demo
├── tests/                       # Unit tests
├── docs/                        # Documentation
│   ├── HPVD_Architecture_Document.md
│   └── HPVD_Technical_Specification.md
├── sources/                     # Source documents
├── requirements.txt             # Dependencies
├── pyproject.toml              # Project configuration
└── README.md
```

## Quick Start

### 1. Create Virtual Environment

```bash
cd HPVD-M22
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Run BM25 Prototype Demo

```bash
python -m src.prototypes.bm25_prototype
```

## Key Components

### Trajectory (60×45 Matrix)
Financial trajectory representing 60 days of evolution across 45 features (R45).

### SparseRegimeIndex
Inverted index for O(1) lookup by regime combination (trend, volatility, structural).

### DenseTrajectoryIndex
FAISS-based index supporting exact and approximate nearest neighbor search.

### HybridDistanceCalculator
Multi-component distance: `D = (0.3×Euclidean + 0.4×Cosine + 0.3×Temporal) × (1 + Regime_Penalty)`

### HPVDEngine
Main orchestrator combining sparse filtering → dense retrieval → hybrid reranking.

## Usage Example

```python
from src.hpvd import HPVDEngine, Trajectory
import numpy as np

# Create engine
engine = HPVDEngine()

# Build index with trajectories
trajectories = [...]  # List of Trajectory objects
engine.build(trajectories)

# Search for similar trajectories
query = Trajectory(
    asset_id="AAPL",
    matrix=np.random.randn(60, 45).astype(np.float32),
    embedding=np.random.randn(256).astype(np.float32),
    trend_regime=1,
    volatility_regime=0,
    structural_regime=1
)

result = engine.search(query, k=25)

# Access results
print(f"Found {result.k_returned} analogs")
print(f"P(H1 up): {result.forecast_h1.p_up:.2%}")
print(f"ACI: {result.aci:.3f}")
```

## Documentation

- [Architecture Document](docs/HPVD_Architecture_Document.md) - High-level design
- [Technical Specification](docs/HPVD_Technical_Specification.md) - Implementation details

## License

MIT License - see LICENSE file for details.

## Project: Kalibry Finance

This is the HPVD component of the Kalibry Financial MVP - a Trajectory Intelligence Engine for short-term financial forecasting.

