"""
Analog Family Formation Module
==============================

Matrix22: Analog Family formation engine that groups candidates
into coherent evolutionary families with explicit uncertainty preservation.

An Analog Family is:
- a coherent group of historical trajectories
- that evolved under compatible structural constraints
- and share evolutionary phase identity
- with explicit uncertainty preserved

It is NOT:
- a cluster in feature space
- a nearest-neighbor list
- a regime label
- a template for action
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np


@dataclass
class FamilyMember:
    """
    Member reference in an Analog Family.
    
    Matrix22: Confidence = structural compatibility, NOT success/outcome.
    """
    trajectory_id: str
    confidence: float  # Structural compatibility score (0-1)


@dataclass
class FamilyCoherence:
    """
    Family-level coherence metrics.
    
    Matrix22: Descriptive only - HPVD does not "fix" weak families.
    
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
    
    Matrix22: Descriptive, not evaluative. Allows downstream to compare/weigh.
    
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
    
    Matrix22: These flags exist to prevent overconfidence, not to gate execution.
    
    Attributes:
        phase_boundary: Family spans phase boundaries
        weak_support: Small family size or high dispersion
        partial_overlap: Family overlaps with others structurally
    """
    phase_boundary: bool = False
    weak_support: bool = False
    partial_overlap: bool = False


@dataclass
class AnalogFamily:
    """
    Analog Family - coherent group of historical trajectories.
    
    Matrix22: This is the primary output structure of HPVD.
    
    Attributes:
        family_id: Purely referential ID, no semantics, stable across replay
        members: List of FamilyMember with trajectory_id and confidence
        coherence: Family-level coherence metrics
        structural_signature: Structural compatibility summary
        uncertainty_flags: Explicit uncertainty annotations
    """
    family_id: str
    members: List[FamilyMember]
    coherence: FamilyCoherence
    structural_signature: StructuralSignature
    uncertainty_flags: UncertaintyFlags


@dataclass
class FamilyFormationConfig:
    """Configuration for family formation"""
    
    # Minimum confidence to be admitted to a family
    admission_threshold: float = 0.3
    
    # Weak support thresholds
    min_family_size: int = 5
    max_dispersion_for_strong: float = 0.3
    min_mean_confidence_for_strong: float = 0.4
    
    # Phase boundary detection
    phase_boundary_diff: int = 1  # Max regime diff to be same phase


class FamilyFormationEngine:
    """
    Engine for forming Analog Families from admitted candidates.
    
    Matrix22:
    - Families are not forced to be compact
    - Families may overlap structurally
    - Families may be small or large
    - HPVD never forces a single family
    - HPVD never merges incompatible groups
    - HPVD never collapses ambiguity
    """
    
    def __init__(self, config: FamilyFormationConfig = None):
        self.config = config or FamilyFormationConfig()
    
    def form_families(
        self,
        admitted_candidates: List[Dict],
        query_regime: Tuple[int, int, int]
    ) -> List[AnalogFamily]:
        """
        Form analog families from admitted candidates.
        
        Args:
            admitted_candidates: List of candidate dicts with:
                - trajectory_id: str
                - confidence: float
                - regime_tuple: Tuple[int, int, int]
                - (optional) geometry_context: Dict
            query_regime: The query's regime tuple (trend, vol, struct)
            
        Returns:
            List of AnalogFamily objects
        """
        if not admitted_candidates:
            return []
        
        # Group by regime tuple
        regime_groups = self._group_by_regime(admitted_candidates)
        
        # Form families from regime groups
        families = []
        family_counter = 0
        
        for regime, members in regime_groups.items():
            if not members:
                continue
            
            family_counter += 1
            family = self._create_family(
                family_id=f"AF_{family_counter:03d}",
                members=members,
                regime=regime,
                query_regime=query_regime,
                all_regimes=set(regime_groups.keys())
            )
            families.append(family)
        
        # Detect partial overlaps between families
        self._detect_partial_overlaps(families)
        
        return families
    
    def _group_by_regime(
        self,
        candidates: List[Dict]
    ) -> Dict[Tuple[int, int, int], List[Dict]]:
        """Group candidates by their regime tuple"""
        regime_groups: Dict[Tuple[int, int, int], List[Dict]] = {}
        
        for candidate in candidates:
            regime = candidate.get('regime_tuple', (0, 0, 0))
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(candidate)
        
        return regime_groups
    
    def _create_family(
        self,
        family_id: str,
        members: List[Dict],
        regime: Tuple[int, int, int],
        query_regime: Tuple[int, int, int],
        all_regimes: Set[Tuple[int, int, int]]
    ) -> AnalogFamily:
        """Create a single AnalogFamily from grouped members"""
        
        # Sort members by confidence (descending)
        members_sorted = sorted(
            members,
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )
        
        # Create FamilyMember objects
        family_members = [
            FamilyMember(
                trajectory_id=m['trajectory_id'],
                confidence=m.get('confidence', 0.0)
            )
            for m in members_sorted
        ]
        
        # Compute coherence
        coherence = self._compute_coherence(members_sorted)
        
        # Compute structural signature
        structural_sig = self._compute_structural_signature(
            regime, members_sorted
        )
        
        # Compute uncertainty flags
        uncertainty = self._compute_uncertainty_flags(
            coherence=coherence,
            candidate_regime=regime,
            query_regime=query_regime,
            all_regimes=all_regimes
        )
        
        return AnalogFamily(
            family_id=family_id,
            members=family_members,
            coherence=coherence,
            structural_signature=structural_sig,
            uncertainty_flags=uncertainty
        )
    
    def _compute_coherence(self, members: List[Dict]) -> FamilyCoherence:
        """Compute coherence metrics for a family"""
        confidences = [m.get('confidence', 0.0) for m in members]
        
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        dispersion = float(np.std(confidences)) if len(confidences) > 1 else 0.0
        
        return FamilyCoherence(
            mean_confidence=mean_conf,
            dispersion=dispersion,
            size=len(members)
        )
    
    def _compute_structural_signature(
        self,
        regime: Tuple[int, int, int],
        members: List[Dict]
    ) -> StructuralSignature:
        """Compute structural signature from regime and geometry context"""
        
        phase_name = self._regime_to_phase_name(regime)
        
        # Extract geometry averages if available
        avg_K = None
        avg_LTV = None
        avg_LVC = None
        
        k_values = []
        ltv_values = []
        lvc_values = []
        
        for m in members:
            geo = m.get('geometry_context', {})
            if 'K' in geo:
                k_values.append(geo['K'])
            if 'LTV' in geo:
                ltv_values.append(geo['LTV'])
            if 'LVC' in geo:
                lvc_values.append(geo['LVC'])
        
        if k_values:
            avg_K = float(np.mean(k_values))
        if ltv_values:
            avg_LTV = float(np.mean(ltv_values))
        if lvc_values:
            avg_LVC = float(np.mean(lvc_values))
        
        return StructuralSignature(
            phase=phase_name,
            avg_K=avg_K,
            avg_LTV=avg_LTV,
            avg_LVC=avg_LVC
        )
    
    def _regime_to_phase_name(self, regime: Tuple[int, int, int]) -> str:
        """Convert regime tuple to descriptive phase name"""
        trend, vol, struct = regime
        
        # Mapping based on regime characteristics
        if trend == 1 and vol == 0 and struct == 1:
            return "stable_expansion"
        elif trend == -1 and vol == 0 and struct == -1:
            return "stable_contraction"
        elif trend == 1 and vol == 1:
            return "volatile_expansion"
        elif trend == -1 and vol == 1:
            return "volatile_contraction"
        elif vol == 1 or struct == 1:
            return "compression_transition"
        elif trend == 0 and vol == 0:
            return "neutral_stable"
        elif trend == 0:
            return "transitional"
        else:
            return "mixed_regime"
    
    def _compute_uncertainty_flags(
        self,
        coherence: FamilyCoherence,
        candidate_regime: Tuple[int, int, int],
        query_regime: Tuple[int, int, int],
        all_regimes: Set[Tuple[int, int, int]]
    ) -> UncertaintyFlags:
        """Compute uncertainty flags for honesty markers"""
        
        # Phase boundary: candidate regime differs significantly from query
        phase_boundary = self._is_phase_boundary(
            candidate_regime, query_regime
        )
        
        # Weak support: small family, high dispersion, or low mean confidence
        weak_support = (
            coherence.size < self.config.min_family_size or
            coherence.dispersion > self.config.max_dispersion_for_strong or
            coherence.mean_confidence < self.config.min_mean_confidence_for_strong
        )
        
        # Partial overlap: detected later in _detect_partial_overlaps
        partial_overlap = False
        
        return UncertaintyFlags(
            phase_boundary=phase_boundary,
            weak_support=weak_support,
            partial_overlap=partial_overlap
        )
    
    def _is_phase_boundary(
        self,
        candidate_regime: Tuple[int, int, int],
        query_regime: Tuple[int, int, int]
    ) -> bool:
        """Check if candidate is at phase boundary relative to query"""
        for c, q in zip(candidate_regime, query_regime):
            if abs(c - q) > self.config.phase_boundary_diff:
                return True
        return False
    
    def _detect_partial_overlaps(self, families: List[AnalogFamily]):
        """
        Detect and flag partial overlaps between families.
        
        Two families overlap if they share similar structural signatures
        but have different phase identities.
        """
        if len(families) < 2:
            return
        
        # Check each pair of families
        for i, family_a in enumerate(families):
            for family_b in families[i+1:]:
                if self._families_overlap(family_a, family_b):
                    family_a.uncertainty_flags.partial_overlap = True
                    family_b.uncertainty_flags.partial_overlap = True
    
    def _families_overlap(
        self,
        family_a: AnalogFamily,
        family_b: AnalogFamily
    ) -> bool:
        """
        Check if two families have structural overlap.
        
        Overlap = similar geometry context but different phases
        """
        sig_a = family_a.structural_signature
        sig_b = family_b.structural_signature
        
        # Different phases
        if sig_a.phase == sig_b.phase:
            return False  # Same phase = not overlap, just same family type
        
        # Check geometry similarity (if available)
        geometry_similar = False
        
        if sig_a.avg_K is not None and sig_b.avg_K is not None:
            k_diff = abs(sig_a.avg_K - sig_b.avg_K)
            if k_diff < 1.0:  # Threshold for "similar"
                geometry_similar = True
        
        if sig_a.avg_LTV is not None and sig_b.avg_LTV is not None:
            ltv_diff = abs(sig_a.avg_LTV - sig_b.avg_LTV)
            if ltv_diff < 0.1:
                geometry_similar = True
        
        return geometry_similar


def compute_family_similarity(
    family_a: AnalogFamily,
    family_b: AnalogFamily
) -> float:
    """
    Compute similarity between two families.
    
    Useful for downstream analysis and family merging decisions
    (which should happen in PMR layer, not HPVD).
    
    Returns:
        Similarity score from 0 (completely different) to 1 (identical)
    """
    # Compare phases
    phase_match = 1.0 if family_a.structural_signature.phase == family_b.structural_signature.phase else 0.0
    
    # Compare coherence
    conf_diff = abs(family_a.coherence.mean_confidence - family_b.coherence.mean_confidence)
    conf_sim = 1.0 - min(conf_diff, 1.0)
    
    # Compare member overlap
    members_a = {m.trajectory_id for m in family_a.members}
    members_b = {m.trajectory_id for m in family_b.members}
    
    if members_a or members_b:
        intersection = len(members_a & members_b)
        union = len(members_a | members_b)
        jaccard = intersection / union if union > 0 else 0.0
    else:
        jaccard = 0.0
    
    # Weighted combination
    similarity = 0.4 * phase_match + 0.3 * conf_sim + 0.3 * jaccard
    
    return similarity
