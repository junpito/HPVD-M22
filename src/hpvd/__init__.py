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
from .embedding import EmbeddingComputer
from .family import (
    FamilyFormationEngine,
    FamilyFormationConfig,
    AnalogFamily,
    FamilyMember,
    FamilyCoherence,
    StructuralSignature,
    UncertaintyFlags,
    compute_family_similarity,
)
from .dna_similarity import (
    DNASimilarityCalculator,
    DNASimilarityConfig,
    extract_phase_from_dna,
    create_synthetic_dna,
)
from .engine import (
    HPVDEngine, HPVDConfig, SearchResult, AnalogResult, HPVD_Output
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
    # Embedding
    "EmbeddingComputer",
    # Legacy search result (backward compat)
    "SearchResult",
    "AnalogResult",
    # Matrix22: Family module (refactored)
    "FamilyFormationEngine",
    "FamilyFormationConfig",
    "HPVD_Output",
    "AnalogFamily",
    "FamilyMember",
    "FamilyCoherence",
    "StructuralSignature",
    "UncertaintyFlags",
    "compute_family_similarity",
    # Matrix22: DNA similarity module
    "DNASimilarityCalculator",
    "DNASimilarityConfig",
    "extract_phase_from_dna",
    "create_synthetic_dna",
    # Synthetic data generator
    "SyntheticDataGenerator",
]

