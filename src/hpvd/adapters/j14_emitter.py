"""
J14 Emitter
============

Wraps raw ``RetrievalResult`` into a ``J14_RetrievalRaw`` envelope.
"""

from .j_file_schemas import J14_RetrievalRaw
from .retrieval_strategy import RetrievalResult


class J14Emitter:
    """Emit ``J14_RetrievalRaw`` from a search result."""

    @staticmethod
    def emit(query_id: str, domain: str, result: RetrievalResult) -> J14_RetrievalRaw:
        return J14_RetrievalRaw(
            query_id=query_id,
            domain=domain,
            candidates=[c.to_dict() for c in result.candidates],
            diagnostics=dict(result.diagnostics),
        )
