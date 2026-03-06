"""
Finance Retrieval Strategy
==========================

Thin wrapper around the existing ``HPVDEngine``.
No HPVD core code is modified — this adapter delegates entirely.

Matrix22: ``score`` = structural compatibility (confidence), NOT outcome probability.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ...engine import HPVDEngine, HPVDConfig, HPVD_Output
from ...trajectory import HPVDInputBundle
from ...family import (
    FamilyCoherence,
    StructuralSignature,
    UncertaintyFlags,
)
from ..retrieval_strategy import (
    FamilyAssignment,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalStrategy,
)


class FinanceRetrievalStrategy(RetrievalStrategy):
    """
    Finance-domain retrieval strategy wrapping ``HPVDEngine``.

    Usage::

        strategy = FinanceRetrievalStrategy(HPVDConfig())
        strategy.build_index(bundles)            # List[HPVDInputBundle]
        result = strategy.search(query_dict, k=25)
        families = strategy.compute_families(result.candidates)
    """

    def __init__(self, config: Optional[HPVDConfig] = None):
        self._config = config or HPVDConfig()
        self._engine = HPVDEngine(self._config)
        # Cache of the most recent HPVD_Output for compute_families()
        self._last_output: Optional[HPVD_Output] = None

    # ------------------------------------------------------------------
    # RetrievalStrategy interface
    # ------------------------------------------------------------------

    @property
    def domain(self) -> str:
        return "finance"

    def build_index(self, corpus: List[HPVDInputBundle]) -> None:
        """Build HPVD indexes from ``HPVDInputBundle`` list."""
        self._engine.build_from_bundles(corpus)

    def search(self, query: Dict[str, Any], k: int = 25) -> RetrievalResult:
        """
        Search for analog trajectories.

        ``query`` must contain an ``"hpvd_input_bundle"`` key whose value
        is an ``HPVDInputBundle`` instance, **or** raw fields
        (``trajectory``, ``dna``, ``geometry_context``, ``metadata``)
        from which a bundle will be constructed.
        """
        bundle = self._resolve_bundle(query)
        # Delegate to HPVDEngine
        output: HPVD_Output = self._engine.search_families(bundle, max_candidates=k * 5)
        self._last_output = output

        # Flatten analog families into RetrievalCandidate list
        candidates: List[RetrievalCandidate] = []
        for family in output.analog_families:
            for member in family.members:
                candidates.append(
                    RetrievalCandidate(
                        candidate_id=member.trajectory_id,
                        score=float(member.confidence),
                        metadata={
                            "family_id": family.family_id,
                            "phase": family.structural_signature.phase,
                        },
                        source_domain="finance",
                    )
                )

        # De-duplicate (a member may appear in only one family, but safety first)
        seen = set()
        unique: List[RetrievalCandidate] = []
        for c in candidates:
            if c.candidate_id not in seen:
                seen.add(c.candidate_id)
                unique.append(c)

        # Sort by score descending
        unique.sort(key=lambda c: c.score, reverse=True)

        return RetrievalResult(
            candidates=unique[:k],
            diagnostics=output.retrieval_diagnostics,
            query_id=output.metadata.get("query_id", ""),
        )

    def compute_families(
        self, candidates: List[RetrievalCandidate]
    ) -> List[FamilyAssignment]:
        """
        Return family assignments.

        For finance, families are already computed inside ``search_families()``,
        so this method converts the cached ``HPVD_Output`` into
        ``FamilyAssignment`` objects.
        """
        if self._last_output is None:
            return []

        # Build a lookup set from the provided candidates for filtering
        candidate_ids = {c.candidate_id for c in candidates}

        assignments: List[FamilyAssignment] = []
        for af in self._last_output.analog_families:
            members = [
                RetrievalCandidate(
                    candidate_id=m.trajectory_id,
                    score=float(m.confidence),
                    metadata={"phase": af.structural_signature.phase},
                    source_domain="finance",
                )
                for m in af.members
                if m.trajectory_id in candidate_ids
            ]
            if not members:
                continue

            assignments.append(
                FamilyAssignment(
                    family_id=af.family_id,
                    members=members,
                    coherence=af.coherence,
                    structural_signature=af.structural_signature,
                    uncertainty_flags=af.uncertainty_flags,
                )
            )
        return assignments

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save underlying HPVDEngine to disk."""
        self._engine.save(path)

    def load(self, path: str) -> None:
        """Load underlying HPVDEngine from disk."""
        self._engine.load(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_bundle(query: Dict[str, Any]) -> HPVDInputBundle:
        """Extract or construct an ``HPVDInputBundle`` from *query* dict."""
        if "hpvd_input_bundle" in query:
            bundle = query["hpvd_input_bundle"]
            if isinstance(bundle, HPVDInputBundle):
                return bundle

        # Construct from raw fields
        trajectory = query.get("trajectory")
        dna = query.get("dna")
        geometry_context = query.get("geometry_context", {})
        metadata = query.get("metadata", {})

        if trajectory is None:
            raise ValueError(
                "Finance query must contain 'hpvd_input_bundle' or 'trajectory' key."
            )
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory, dtype=np.float32)
        if dna is None:
            dna = np.zeros(16, dtype=np.float32)
        elif not isinstance(dna, np.ndarray):
            dna = np.array(dna, dtype=np.float32)

        return HPVDInputBundle(
            trajectory=trajectory.astype(np.float32),
            dna=dna.astype(np.float32),
            geometry_context=geometry_context,
            metadata=metadata,
        )
