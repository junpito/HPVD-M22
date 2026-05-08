"""
HPVD — Knowledge & Document Retrieval Engine
==============================================

Retrieval engine for the Manithy pipeline.  HPVD provides two retrieval
strategies:

    - ``KnowledgeRetrievalStrategy``  — sector-based knowledge object retrieval
    - ``DocumentRetrievalStrategy``   — semantic vector search over document chunks

Version: 2.0.0-alpha1  (Manithy refactor)
"""

__version__ = "2.0.0a1"
__author__ = "Kalibry Team"

from .adapters.family_types import (
    FamilyCoherence,
    StructuralSignature,
    UncertaintyFlags,
)

from .adapters import (
    RetrievalStrategy,
    RetrievalCandidate,
    RetrievalResult,
    FamilyAssignment,
    DocumentRetrievalStrategy,
    DocumentChunk,
    DocumentRetrievalConfig,
    KnowledgeRetrievalStrategy,
)

__all__ = [
    # Family types
    "FamilyCoherence",
    "StructuralSignature",
    "UncertaintyFlags",
    # Adapter layer
    "RetrievalStrategy",
    "RetrievalCandidate",
    "RetrievalResult",
    "FamilyAssignment",
    "DocumentRetrievalStrategy",
    "DocumentChunk",
    "DocumentRetrievalConfig",
    "KnowledgeRetrievalStrategy",
]
