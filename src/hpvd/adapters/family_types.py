"""
Family Types — Shared Dataclasses
===================================

Lightweight dataclasses reused across retrieval strategies for
coherence metrics, structural signatures, and uncertainty flags.

These are the only types retained from the original family module;
the ``FamilyFormationEngine`` and ``AnalogFamily`` classes have been
removed as part of the Manithy v1 refactoring.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FamilyCoherence:
    """
    Family-level coherence metrics.

    Attributes:
        mean_confidence: Average confidence of members
        dispersion: Standard deviation of confidences (high = less coherent)
        size: Number of members in the family
    """
    mean_confidence: float
    dispersion: float  # Standard deviation of confidences
    size: int


@dataclass
class StructuralSignature:
    """
    Structural compatibility summary for a family.

    Attributes:
        phase: Phase name (e.g., "stable_expansion", "compression_transition")
        avg_K: Average curvature (if available from geometry_context)
        avg_LTV: Average LTV (if available)
        avg_LVC: Average LVC (if available)
    """
    phase: str  # e.g., "stable_expansion", "compression_transition"
    avg_K: Optional[float] = None
    avg_LTV: Optional[float] = None
    avg_LVC: Optional[float] = None


@dataclass
class UncertaintyFlags:
    """
    Explicit honesty markers to prevent overconfidence downstream.

    Attributes:
        phase_boundary: Family spans phase boundaries
        weak_support: Small family size or high dispersion
        partial_overlap: Family overlaps with others structurally
    """
    phase_boundary: bool = False
    weak_support: bool = False
    partial_overlap: bool = False
