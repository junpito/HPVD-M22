"""
HPVD Adapters Layer
===================

Strategy pattern for multi-domain retrieval (finance, document, etc.)
with J-file schema adapters (J13→J16) and a unified pipeline engine.

This layer wraps the HPVD core without modifying it.
"""

from .retrieval_strategy import (
    RetrievalCandidate,
    RetrievalResult,
    FamilyAssignment,
    RetrievalStrategy,
)
from .j_file_schemas import (
    J13_PostCoreQuery,
    J14_RetrievalRaw,
    J15_PhaseFilteredSet,
    J16_AnalogFamilyAssignment,
)
from .strategy_dispatcher import StrategyDispatcher
from .j13_adapter import J13Adapter
from .j14_emitter import J14Emitter
from .j15_emitter import J15Emitter
from .j16_emitter import J16Emitter
from .pipeline_engine import HPVDPipelineEngine, PipelineOutput

from .strategies import FinanceRetrievalStrategy, DocumentRetrievalStrategy
from .strategies.document_strategy import DocumentChunk, DocumentRetrievalConfig

__all__ = [
    # ABC + common types
    "RetrievalCandidate",
    "RetrievalResult",
    "FamilyAssignment",
    "RetrievalStrategy",
    # J-file schemas
    "J13_PostCoreQuery",
    "J14_RetrievalRaw",
    "J15_PhaseFilteredSet",
    "J16_AnalogFamilyAssignment",
    # Dispatcher
    "StrategyDispatcher",
    # J-file adapters
    "J13Adapter",
    "J14Emitter",
    "J15Emitter",
    "J16Emitter",
    # Pipeline
    "HPVDPipelineEngine",
    "PipelineOutput",
    # Strategies
    "FinanceRetrievalStrategy",
    "DocumentRetrievalStrategy",
    "DocumentChunk",
    "DocumentRetrievalConfig",
]
