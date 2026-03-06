"""
Retrieval Strategy — Abstract Base Class & Common Types
========================================================

Domain-agnostic output types that all strategies share, plus the ABC
that every concrete strategy must implement.

Matrix22: ``RetrievalCandidate.score`` is **calibrated similarity**
(structural/semantic compatibility), NOT outcome probability.
Outcome-blind principle is preserved across all domains.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..family import (
    FamilyCoherence,
    StructuralSignature,
    UncertaintyFlags,
)


# ---------------------------------------------------------------------------
# Common output types
# ---------------------------------------------------------------------------

@dataclass
class RetrievalCandidate:
    """
    Domain-agnostic retrieval candidate.

    Attributes:
        candidate_id: Unique identifier (trajectory_id for finance, chunk_id for document).
        score: Calibrated similarity in [0, 1]. Higher = more similar.
                Maps to ``confidence`` in finance and ``cosine_similarity`` in document.
        metadata: Arbitrary domain-specific metadata (e.g., regime, topic, doc_type).
        source_domain: Domain tag (``"finance"`` or ``"document"``).
    """

    candidate_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_domain: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": float(self.score),
            "metadata": dict(self.metadata),
            "source_domain": self.source_domain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalCandidate":
        return cls(
            candidate_id=data["candidate_id"],
            score=float(data["score"]),
            metadata=data.get("metadata", {}),
            source_domain=data.get("source_domain", ""),
        )


@dataclass
class RetrievalResult:
    """
    Result of a strategy ``search()`` call.

    Attributes:
        candidates: Ordered list of ``RetrievalCandidate`` (best first).
        diagnostics: Strategy-specific diagnostic counters.
        query_id: Identifier linking back to the originating query.
    """

    candidates: List[RetrievalCandidate]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    query_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "diagnostics": dict(self.diagnostics),
            "query_id": self.query_id,
        }


@dataclass
class FamilyAssignment:
    """
    Family assignment produced by ``compute_families()``.

    Reuses ``FamilyCoherence``, ``StructuralSignature``, and
    ``UncertaintyFlags`` from the HPVD core ``family`` module so that
    the output contract is identical across domains.
    """

    family_id: str
    members: List[RetrievalCandidate]
    coherence: FamilyCoherence
    structural_signature: StructuralSignature
    uncertainty_flags: UncertaintyFlags

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "members": [m.to_dict() for m in self.members],
            "coherence": {
                "mean_confidence": float(self.coherence.mean_confidence),
                "dispersion": float(self.coherence.dispersion),
                "size": int(self.coherence.size),
            },
            "structural_signature": {
                "phase": self.structural_signature.phase,
                "avg_K": (
                    float(self.structural_signature.avg_K)
                    if self.structural_signature.avg_K is not None
                    else None
                ),
                "avg_LTV": (
                    float(self.structural_signature.avg_LTV)
                    if self.structural_signature.avg_LTV is not None
                    else None
                ),
                "avg_LVC": (
                    float(self.structural_signature.avg_LVC)
                    if self.structural_signature.avg_LVC is not None
                    else None
                ),
            },
            "uncertainty_flags": {
                "phase_boundary": self.uncertainty_flags.phase_boundary,
                "weak_support": self.uncertainty_flags.weak_support,
                "partial_overlap": self.uncertainty_flags.partial_overlap,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FamilyAssignment":
        members = [RetrievalCandidate.from_dict(m) for m in data["members"]]
        coh = data["coherence"]
        coherence = FamilyCoherence(
            mean_confidence=float(coh["mean_confidence"]),
            dispersion=float(coh["dispersion"]),
            size=int(coh["size"]),
        )
        sig = data["structural_signature"]
        structural_signature = StructuralSignature(
            phase=sig["phase"],
            avg_K=float(sig["avg_K"]) if sig.get("avg_K") is not None else None,
            avg_LTV=float(sig["avg_LTV"]) if sig.get("avg_LTV") is not None else None,
            avg_LVC=float(sig["avg_LVC"]) if sig.get("avg_LVC") is not None else None,
        )
        uf = data["uncertainty_flags"]
        uncertainty_flags = UncertaintyFlags(
            phase_boundary=bool(uf["phase_boundary"]),
            weak_support=bool(uf["weak_support"]),
            partial_overlap=bool(uf["partial_overlap"]),
        )
        return cls(
            family_id=data["family_id"],
            members=members,
            coherence=coherence,
            structural_signature=structural_signature,
            uncertainty_flags=uncertainty_flags,
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class RetrievalStrategy(ABC):
    """
    Abstract retrieval strategy.

    Every concrete domain strategy must implement:
        - ``domain`` property  (e.g. ``"finance"``, ``"document"``)
        - ``build_index(corpus)``
        - ``search(query, k) → RetrievalResult``
        - ``compute_families(candidates) → List[FamilyAssignment]``
    """

    @property
    @abstractmethod
    def domain(self) -> str:
        """Return the canonical domain name (e.g. ``'finance'``, ``'document'``)."""
        ...

    @abstractmethod
    def build_index(self, corpus: Any) -> None:
        """Build the search index from a domain-specific corpus."""
        ...

    @abstractmethod
    def search(self, query: Dict[str, Any], k: int = 25) -> RetrievalResult:
        """Execute a search and return ordered candidates."""
        ...

    @abstractmethod
    def compute_families(
        self, candidates: List[RetrievalCandidate]
    ) -> List[FamilyAssignment]:
        """Group candidates into coherent families."""
        ...
