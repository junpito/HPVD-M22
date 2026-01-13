"""
HPVD - Hybrid Probabilistic Vector Database
============================================

A high-performance vector search engine for finding historical analogs
in financial trajectory data.

Components:
- SparseRegimeIndex: Regime-based inverted index for filtering
- DenseTrajectoryIndex: FAISS-based dense vector search
- HPVDEngine: Main search engine combining sparse + dense
- HybridDistanceCalculator: Multi-component distance metrics

Version: 1.0.0-MVP
Project: Kalibry Finance
"""

__version__ = "1.0.0a1"  # MVP version
__author__ = "Kalibry Team"

from .trajectory import Trajectory, HPVDInputBundle
from .synthetic_data_generator import SyntheticDataGenerator
from .sparse_index import SparseRegimeIndex
from .dense_index import DenseTrajectoryIndex, DenseIndexConfig
from .distance import HybridDistanceCalculator, DistanceConfig
from .engine import (
    HPVDEngine, HPVDConfig, SearchResult, AnalogResult,
    HPVD_Output, AnalogFamily, FamilyMember, FamilyCoherence,
    StructuralSignature, UncertaintyFlags
)

__all__ = [
    # Core classes
    "Trajectory",
    "HPVDInputBundle",
    "SparseRegimeIndex", 
    "DenseTrajectoryIndex",
    "DenseIndexConfig",
    "HybridDistanceCalculator",
    "DistanceConfig",
    "HPVDEngine",
    "HPVDConfig",
    # Legacy search result (backward compat)
    "SearchResult",
    "AnalogResult",
    # Matrix22: New family-based output
    "HPVD_Output",
    "AnalogFamily",
    "FamilyMember",
    "FamilyCoherence",
    "StructuralSignature",
    "UncertaintyFlags",
    # Synthetic data generator
    "SyntheticDataGenerator",
]

