"""
Hybrid Distance Calculator
==========================

Multi-component distance metrics for trajectory similarity.
Combines Euclidean, Cosine, and Temporal-weighted distances.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


@dataclass
class DistanceConfig:
    """Configuration for hybrid distance computation"""
    
    # Component weights (must sum to 1.0)
    weight_euclidean: float = 0.3
    weight_cosine: float = 0.4
    weight_temporal: float = 0.3
    
    # Regime penalty
    regime_penalty_weight: float = 0.2
    
    # Temporal decay
    temporal_decay: float = 0.95   # Recent days weighted more
    
    def __post_init__(self):
        total = self.weight_euclidean + self.weight_cosine + self.weight_temporal
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


class HybridDistanceCalculator:
    """
    Compute hybrid distance between trajectory matrices
    
    Components:
    1. Euclidean distance (flattened vectors)
    2. Cosine distance (angular similarity)
    3. Temporal-weighted distance (recent days weighted more)
    4. Regime penalty (mismatch penalty)
    """
    
    def __init__(self, config: DistanceConfig = None):
        self.config = config or DistanceConfig()
        
        # Precompute temporal weights for 60-day window
        self.temporal_weights = self._compute_temporal_weights(60)
    
    def _compute_temporal_weights(self, window: int) -> np.ndarray:
        """
        Compute exponential decay weights
        
        w_t = decay^(T-1-t) for t = 0..T-1
        More recent days (higher t) get higher weight
        """
        decay = self.config.temporal_decay
        weights = np.array([
            decay ** (window - 1 - t) for t in range(window)
        ])
        return weights / weights.sum()  # Normalize to sum=1
    
    def euclidean_distance(self,
                           matrix_a: np.ndarray,
                           matrix_b: np.ndarray) -> float:
        """
        Euclidean distance between flattened matrices
        
        d_euc = ||flatten(A) - flatten(B)||_2
        """
        flat_a = matrix_a.flatten()
        flat_b = matrix_b.flatten()
        return float(np.linalg.norm(flat_a - flat_b))
    
    def cosine_distance(self,
                        matrix_a: np.ndarray,
                        matrix_b: np.ndarray) -> float:
        """
        Cosine distance between flattened matrices
        
        d_cos = 1 - cos(θ) = 1 - (A·B)/(||A|| ||B||)
        """
        flat_a = matrix_a.flatten()
        flat_b = matrix_b.flatten()
        
        dot = np.dot(flat_a, flat_b)
        norm_a = np.linalg.norm(flat_a)
        norm_b = np.linalg.norm(flat_b)
        
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 1.0  # Maximum distance for zero vectors
        
        cosine_sim = dot / (norm_a * norm_b)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        return float(1.0 - cosine_sim)
    
    def temporal_weighted_distance(self,
                                    matrix_a: np.ndarray,
                                    matrix_b: np.ndarray) -> float:
        """
        Time-weighted L2 distance
        
        d_temp = Σ_t w_t * ||row_t(A) - row_t(B)||_2
        
        Recent days contribute more to the distance
        """
        # Per-day L2 distance
        day_distances = np.linalg.norm(matrix_a - matrix_b, axis=1)  # (60,)
        
        # Weighted sum
        return float(np.dot(self.temporal_weights, day_distances))
    
    def regime_match_score(self,
                           regime_a: Tuple[int, int, int],
                           regime_b: Tuple[int, int, int]) -> float:
        """
        Compute regime match score
        
        Returns:
            Score from 0 (no match) to 1 (exact match)
        """
        scores = []
        for va, vb in zip(regime_a, regime_b):
            diff = abs(va - vb)
            if diff == 0:
                scores.append(1.0)    # Exact match
            elif diff == 1:
                scores.append(0.5)    # Adjacent regime
            else:
                scores.append(0.0)    # Far regime
        
        return float(np.mean(scores))
    
    def compute(self,
                matrix_a: np.ndarray,
                matrix_b: np.ndarray,
                regime_a: Tuple[int, int, int],
                regime_b: Tuple[int, int, int]) -> Tuple[float, Dict]:
        """
        Compute full hybrid distance
        
        Args:
            matrix_a: (60, 45) query trajectory matrix
            matrix_b: (60, 45) candidate trajectory matrix
            regime_a: Query regime tuple (trend, vol, struct)
            regime_b: Candidate regime tuple
            
        Returns:
            (total_distance, components_dict)
        """
        # Compute component distances
        d_euc = self.euclidean_distance(matrix_a, matrix_b)
        d_cos = self.cosine_distance(matrix_a, matrix_b)
        d_temp = self.temporal_weighted_distance(matrix_a, matrix_b)
        
        # Regime match
        regime_match = self.regime_match_score(regime_a, regime_b)
        
        # Normalize distances to [0, 1] range (approximately)
        d_euc_norm = d_euc / (np.sqrt(2700) * 2)
        d_cos_norm = d_cos / 2.0
        d_temp_norm = d_temp / (np.sqrt(45) * 2)
        
        # Clamp to [0, 1]
        d_euc_norm = min(d_euc_norm, 1.0)
        d_cos_norm = min(d_cos_norm, 1.0)
        d_temp_norm = min(d_temp_norm, 1.0)
        
        # Weighted combination
        base_distance = (
            self.config.weight_euclidean * d_euc_norm +
            self.config.weight_cosine * d_cos_norm +
            self.config.weight_temporal * d_temp_norm
        )
        
        # Apply regime penalty
        regime_penalty = (1.0 - regime_match) * self.config.regime_penalty_weight
        total_distance = base_distance * (1.0 + regime_penalty)
        
        # Return components for debugging/explainability
        components = {
            'euclidean_raw': d_euc,
            'euclidean_norm': d_euc_norm,
            'cosine_raw': d_cos,
            'cosine_norm': d_cos_norm,
            'temporal_raw': d_temp,
            'temporal_norm': d_temp_norm,
            'base_distance': base_distance,
            'regime_match': regime_match,
            'regime_penalty': regime_penalty,
            'total_distance': total_distance
        }
        
        return total_distance, components
    
    def feature_level_distance(self,
                               matrix_a: np.ndarray,
                               matrix_b: np.ndarray) -> np.ndarray:
        """
        Compute per-feature distance for explainability
        
        Returns:
            (45,) array of time-weighted distance per feature
        """
        diff = np.abs(matrix_a - matrix_b)  # (60, 45)
        weighted_diff = diff * self.temporal_weights.reshape(-1, 1)
        
        return weighted_diff.sum(axis=0)  # Sum across time: (45,)

