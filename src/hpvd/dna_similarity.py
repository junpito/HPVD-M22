"""
DNA Similarity Module
=====================

Matrix22: Compute similarity between Cognitive DNA vectors.

Cognitive DNA encodes evolutionary phase identity:
- Phase (expansion / contraction / transition)
- Stability vs instability
- Cyclicity vs drift
- Regime identity

What DNA does NOT encode:
- Outcomes
- Direction
- Success / failure

Two trajectories may differ in values but share DNA similarity.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np


@dataclass
class DNASimilarityConfig:
    """Configuration for DNA similarity computation"""
    
    # Weights for different similarity components
    weight_cosine: float = 0.4
    weight_euclidean: float = 0.3
    weight_phase_proximity: float = 0.3
    
    # Phase proximity thresholds
    phase_match_threshold: float = 0.8  # High similarity = same phase
    phase_adjacent_threshold: float = 0.5  # Medium similarity = adjacent phase
    
    # Normalization
    normalize_dna: bool = True
    
    def __post_init__(self):
        total = self.weight_cosine + self.weight_euclidean + self.weight_phase_proximity
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


class DNASimilarityCalculator:
    """
    Calculate similarity between Cognitive DNA vectors.
    
    Matrix22: DNA similarity is used by HPVD to find trajectories
    that share evolutionary phase identity, independent of outcomes.
    
    Methods:
    - cosine_similarity: Angular similarity between DNA vectors
    - euclidean_distance: L2 distance (normalized)
    - phase_proximity: Estimate phase compatibility
    - compute: Full multi-component similarity
    """
    
    def __init__(self, config: DNASimilarityConfig = None):
        self.config = config or DNASimilarityConfig()
    
    def cosine_similarity(self, dna_a: np.ndarray, dna_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two DNA vectors.
        
        cos(θ) = (A·B) / (||A|| ||B||)
        
        Returns:
            Similarity from -1 (opposite) to 1 (identical)
        """
        # Normalize if configured
        if self.config.normalize_dna:
            dna_a = self._normalize(dna_a)
            dna_b = self._normalize(dna_b)
        
        dot = np.dot(dna_a, dna_b)
        norm_a = np.linalg.norm(dna_a)
        norm_b = np.linalg.norm(dna_b)
        
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0  # Undefined for zero vectors
        
        similarity = dot / (norm_a * norm_b)
        return float(np.clip(similarity, -1.0, 1.0))
    
    def cosine_distance(self, dna_a: np.ndarray, dna_b: np.ndarray) -> float:
        """
        Compute cosine distance (1 - cosine_similarity).
        
        Returns:
            Distance from 0 (identical) to 2 (opposite)
        """
        return 1.0 - self.cosine_similarity(dna_a, dna_b)
    
    def euclidean_distance(self, dna_a: np.ndarray, dna_b: np.ndarray) -> float:
        """
        Compute normalized Euclidean distance between DNA vectors.
        
        Returns:
            Distance from 0 (identical) to 1 (normalized max)
        """
        # Normalize if configured
        if self.config.normalize_dna:
            dna_a = self._normalize(dna_a)
            dna_b = self._normalize(dna_b)
        
        raw_dist = np.linalg.norm(dna_a - dna_b)
        
        # Normalize by max possible distance (sqrt(2) for unit vectors)
        max_dist = np.sqrt(2) * np.sqrt(len(dna_a))
        normalized_dist = raw_dist / max_dist if max_dist > 0 else 0.0
        
        return float(min(normalized_dist, 1.0))
    
    def euclidean_similarity(self, dna_a: np.ndarray, dna_b: np.ndarray) -> float:
        """
        Compute Euclidean similarity (1 - euclidean_distance).
        
        Returns:
            Similarity from 0 (far) to 1 (identical)
        """
        return 1.0 - self.euclidean_distance(dna_a, dna_b)
    
    def phase_proximity(self, dna_a: np.ndarray, dna_b: np.ndarray) -> float:
        """
        Estimate phase proximity between two DNA vectors.
        
        Phase proximity measures how likely two trajectories are
        in the same evolutionary phase, based on DNA structure.
        
        Matrix22: This is a structural estimate, NOT a probability.
        
        Returns:
            Proximity score from 0 (different phase) to 1 (same phase)
        """
        # Normalize if configured
        if self.config.normalize_dna:
            dna_a = self._normalize(dna_a)
            dna_b = self._normalize(dna_b)
        
        # Method 1: Correlation-based phase matching
        if len(dna_a) > 1:
            correlation = np.corrcoef(dna_a, dna_b)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 1.0 if abs(dna_a[0] - dna_b[0]) < 0.1 else 0.0
        
        # Method 2: Sign agreement (for phase indicators)
        sign_a = np.sign(dna_a)
        sign_b = np.sign(dna_b)
        sign_agreement = np.mean(sign_a == sign_b)
        
        # Method 3: Magnitude ratio (similar magnitude = same phase)
        mag_a = np.linalg.norm(dna_a)
        mag_b = np.linalg.norm(dna_b)
        if mag_a > 0 and mag_b > 0:
            mag_ratio = min(mag_a, mag_b) / max(mag_a, mag_b)
        else:
            mag_ratio = 1.0 if mag_a == mag_b else 0.0
        
        # Combine methods
        proximity = 0.4 * max(0, correlation) + 0.3 * sign_agreement + 0.3 * mag_ratio
        
        return float(np.clip(proximity, 0.0, 1.0))
    
    def compute(
        self,
        dna_a: np.ndarray,
        dna_b: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute full multi-component DNA similarity.
        
        Args:
            dna_a: First DNA vector (K-dimensional)
            dna_b: Second DNA vector (K-dimensional)
            
        Returns:
            (similarity_score, components_dict)
            - similarity_score: Combined similarity from 0 to 1
            - components_dict: Individual component scores
        """
        # Validate dimensions
        if dna_a.shape != dna_b.shape:
            raise ValueError(f"DNA shapes must match: {dna_a.shape} vs {dna_b.shape}")
        
        # Compute individual components
        cos_sim = self.cosine_similarity(dna_a, dna_b)
        euc_sim = self.euclidean_similarity(dna_a, dna_b)
        phase_prox = self.phase_proximity(dna_a, dna_b)
        
        # Combine with weights
        total_similarity = (
            self.config.weight_cosine * cos_sim +
            self.config.weight_euclidean * euc_sim +
            self.config.weight_phase_proximity * phase_prox
        )
        
        # Components for debugging/explainability
        components = {
            'cosine_similarity': cos_sim,
            'cosine_distance': 1.0 - cos_sim,
            'euclidean_similarity': euc_sim,
            'euclidean_distance': 1.0 - euc_sim,
            'phase_proximity': phase_prox,
            'total_similarity': total_similarity,
            'total_distance': 1.0 - total_similarity,
        }
        
        return total_similarity, components
    
    def compute_distance(
        self,
        dna_a: np.ndarray,
        dna_b: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute DNA distance (1 - similarity).
        
        Returns:
            (distance, components_dict)
        """
        similarity, components = self.compute(dna_a, dna_b)
        return 1.0 - similarity, components
    
    def classify_phase_relationship(
        self,
        dna_a: np.ndarray,
        dna_b: np.ndarray
    ) -> str:
        """
        Classify the phase relationship between two DNA vectors.
        
        Matrix22: This is descriptive classification, NOT a decision.
        
        Returns:
            One of: "same_phase", "adjacent_phase", "different_phase"
        """
        phase_prox = self.phase_proximity(dna_a, dna_b)
        
        if phase_prox >= self.config.phase_match_threshold:
            return "same_phase"
        elif phase_prox >= self.config.phase_adjacent_threshold:
            return "adjacent_phase"
        else:
            return "different_phase"
    
    def batch_similarity(
        self,
        query_dna: np.ndarray,
        candidate_dnas: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between query DNA and multiple candidates.
        
        Args:
            query_dna: Single DNA vector (K,)
            candidate_dnas: Array of DNA vectors (N, K)
            
        Returns:
            Array of similarities (N,)
        """
        similarities = np.zeros(len(candidate_dnas))
        
        for i, candidate_dna in enumerate(candidate_dnas):
            similarity, _ = self.compute(query_dna, candidate_dna)
            similarities[i] = similarity
        
        return similarities
    
    def _normalize(self, dna: np.ndarray) -> np.ndarray:
        """Normalize DNA vector to unit length"""
        norm = np.linalg.norm(dna)
        if norm < 1e-9:
            return dna
        return dna / norm


def extract_phase_from_dna(dna: np.ndarray) -> Dict[str, float]:
    """
    Extract interpretable phase information from DNA vector.
    
    This is a simplified extraction - real implementation would
    depend on how DNA is constructed in the geometry module.
    
    Args:
        dna: Cognitive DNA vector (K-dimensional)
        
    Returns:
        Dictionary with phase descriptors
    """
    # Simplified phase extraction (placeholder)
    # In real implementation, this would decode the DNA structure
    
    magnitude = float(np.linalg.norm(dna))
    mean_val = float(np.mean(dna))
    std_val = float(np.std(dna))
    
    # Infer phase characteristics (simplified heuristics)
    is_stable = std_val < 0.3
    is_expanding = mean_val > 0.1
    is_contracting = mean_val < -0.1
    
    if is_stable and is_expanding:
        phase_label = "stable_expansion"
    elif is_stable and is_contracting:
        phase_label = "stable_contraction"
    elif is_stable:
        phase_label = "stable_neutral"
    elif is_expanding:
        phase_label = "volatile_expansion"
    elif is_contracting:
        phase_label = "volatile_contraction"
    else:
        phase_label = "transitional"
    
    return {
        'magnitude': magnitude,
        'mean': mean_val,
        'std': std_val,
        'phase_label': phase_label,
        'is_stable': is_stable,
        'is_expanding': is_expanding,
        'is_contracting': is_contracting,
    }


def create_synthetic_dna(
    regime_id: str,
    dim: int = 16,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create synthetic DNA vector for a given regime.
    
    Useful for testing. In production, DNA comes from geometry module.
    
    Args:
        regime_id: Regime identifier (R1, R2, R3, R4, R5, R6)
        dim: DNA dimension (default 16)
        seed: Random seed for reproducibility
        
    Returns:
        DNA vector (dim,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base patterns for each regime
    regime_patterns = {
        'R1': {'mean': 0.5, 'std': 0.1, 'sign': 1},    # Stable expansion
        'R2': {'mean': -0.5, 'std': 0.1, 'sign': -1},  # Stable contraction
        'R3': {'mean': 0.0, 'std': 0.3, 'sign': 1},    # Compression
        'R4': {'mean': 0.0, 'std': 0.5, 'sign': 0},    # Transitional
        'R5': {'mean': 0.3, 'std': 0.4, 'sign': 1},    # Structural stress
        'R6': {'mean': 0.0, 'std': 0.8, 'sign': 0},    # Novel/unseen
    }
    
    pattern = regime_patterns.get(regime_id, regime_patterns['R4'])
    
    # Generate DNA with regime characteristics
    base = np.random.randn(dim) * pattern['std'] + pattern['mean']
    
    # Add regime-specific structure
    if pattern['sign'] != 0:
        base[:dim//2] *= pattern['sign']
    
    return base.astype(np.float32)
