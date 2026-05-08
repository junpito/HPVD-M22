"""
HPVD Retrieval Strategies
=========================

Concrete strategy implementations for different domains.
"""

from .document_strategy import DocumentRetrievalStrategy
from .knowledge_strategy import KnowledgeRetrievalStrategy

__all__ = [
    "DocumentRetrievalStrategy",
    "KnowledgeRetrievalStrategy",
]
