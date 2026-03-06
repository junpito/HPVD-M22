"""
J15 Emitter
============

Applies a phase / topic filter to ``RetrievalResult`` candidates and
splits them into accepted / rejected sets as ``J15_PhaseFilteredSet``.
"""

from typing import Any, Callable, Dict, List, Optional

from .j_file_schemas import J15_PhaseFilteredSet
from .retrieval_strategy import RetrievalCandidate, RetrievalResult


class J15Emitter:
    """Emit ``J15_PhaseFilteredSet`` by applying a filter function."""

    @staticmethod
    def emit(
        query_id: str,
        result: RetrievalResult,
        filter_fn: Optional[Callable[[RetrievalCandidate], bool]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> J15_PhaseFilteredSet:
        """
        Parameters
        ----------
        query_id : str
        result : RetrievalResult
        filter_fn : callable, optional
            Predicate returning ``True`` for accepted candidates.
            If ``None`` all candidates are accepted.
        filter_criteria : dict, optional
            Description of what filter was applied (stored in the J15 envelope).
        """
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for c in result.candidates:
            if filter_fn is None or filter_fn(c):
                accepted.append(c.to_dict())
            else:
                rejected.append(c.to_dict())

        return J15_PhaseFilteredSet(
            query_id=query_id,
            accepted=accepted,
            rejected=rejected,
            filter_criteria=filter_criteria or {},
        )
