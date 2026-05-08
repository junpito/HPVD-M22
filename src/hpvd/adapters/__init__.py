"""
HPVD Adapters Layer
===================

Strategy pattern for multi-domain retrieval (knowledge, document).
"""

from .retrieval_strategy import (
    RetrievalCandidate,
    RetrievalResult,
    FamilyAssignment,
    RetrievalStrategy,
)

from .strategies import DocumentRetrievalStrategy, KnowledgeRetrievalStrategy
from .strategies.document_strategy import DocumentChunk, DocumentRetrievalConfig

__all__ = [
    # ABC + common types
    "RetrievalCandidate",
    "RetrievalResult",
    "FamilyAssignment",
    "RetrievalStrategy",
    # Strategies
    "DocumentRetrievalStrategy",
    "DocumentChunk",
    "DocumentRetrievalConfig",
    "KnowledgeRetrievalStrategy",
]
