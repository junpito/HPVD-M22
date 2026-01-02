"""
Sparse Regime Index
===================

Inverted index for regime-based trajectory filtering.
Enables O(1) lookup of trajectories by regime combination.
"""

from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, List
import pickle


class SparseRegimeIndex:
    """
    Inverted index for regime-based trajectory filtering
    
    Enables O(1) lookup of trajectories by regime combination.
    
    Index Structure:
    - Primary: (trend, vol, struct) → Set[trajectory_id]
    - Asset: asset_id → Set[trajectory_id]
    - Asset Class: asset_class → Set[trajectory_id]
    """
    
    def __init__(self):
        # Primary index: regime tuple → trajectory IDs
        self.regime_index: Dict[Tuple[int, int, int], Set[str]] = defaultdict(set)
        
        # Asset index: asset_id → trajectory IDs
        self.asset_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Asset class index: asset_class → trajectory IDs
        self.asset_class_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Reverse lookup: trajectory_id → regime tuple
        self.trajectory_regimes: Dict[str, Tuple[int, int, int]] = {}
        
        # Statistics
        self.total_count: int = 0
    
    def add(self, 
            trajectory_id: str,
            trend: int,
            volatility: int,
            structural: int,
            asset_id: str,
            asset_class: str = "equity"):
        """
        Add trajectory to all indexes
        
        Args:
            trajectory_id: Unique trajectory identifier
            trend: Trend regime (-1, 0, +1)
            volatility: Volatility regime (-1, 0, +1)
            structural: Structural regime (-1, 0, +1)
            asset_id: Asset ticker
            asset_class: Asset category
        """
        # Validate regimes
        assert trend in [-1, 0, 1], f"Invalid trend: {trend}"
        assert volatility in [-1, 0, 1], f"Invalid volatility: {volatility}"
        assert structural in [-1, 0, 1], f"Invalid structural: {structural}"
        
        regime_key = (trend, volatility, structural)
        
        # Add to indexes
        self.regime_index[regime_key].add(trajectory_id)
        self.asset_index[asset_id].add(trajectory_id)
        self.asset_class_index[asset_class].add(trajectory_id)
        self.trajectory_regimes[trajectory_id] = regime_key
        
        self.total_count += 1
    
    def remove(self, trajectory_id: str):
        """Remove trajectory from all indexes"""
        if trajectory_id not in self.trajectory_regimes:
            return
        
        regime_key = self.trajectory_regimes[trajectory_id]
        
        # Remove from regime index
        if trajectory_id in self.regime_index[regime_key]:
            self.regime_index[regime_key].remove(trajectory_id)
        
        # Remove from other indexes
        for asset_set in self.asset_index.values():
            asset_set.discard(trajectory_id)
        for class_set in self.asset_class_index.values():
            class_set.discard(trajectory_id)
        
        del self.trajectory_regimes[trajectory_id]
        self.total_count -= 1
    
    def filter_by_regime(self,
                         trend: Optional[int] = None,
                         volatility: Optional[int] = None,
                         structural: Optional[int] = None,
                         allow_adjacent: bool = True) -> Set[str]:
        """
        Filter trajectories by regime constraints
        
        Args:
            trend: Target trend regime or None for any
            volatility: Target volatility regime or None
            structural: Target structural regime or None
            allow_adjacent: Include adjacent regimes (±1)
            
        Returns:
            Set of matching trajectory IDs
        """
        result = set()
        
        for regime_key, trajectories in self.regime_index.items():
            k_trend, k_vol, k_struct = regime_key
            match = True
            
            # Check trend
            if trend is not None:
                if allow_adjacent:
                    match = match and abs(k_trend - trend) <= 1
                else:
                    match = match and k_trend == trend
            
            # Check volatility
            if volatility is not None:
                if allow_adjacent:
                    match = match and abs(k_vol - volatility) <= 1
                else:
                    match = match and k_vol == volatility
            
            # Check structural
            if structural is not None:
                if allow_adjacent:
                    match = match and abs(k_struct - structural) <= 1
                else:
                    match = match and k_struct == structural
            
            if match:
                result.update(trajectories)
        
        return result
    
    def filter_by_asset(self, asset_ids: List[str]) -> Set[str]:
        """Filter by specific assets"""
        result = set()
        for asset_id in asset_ids:
            result.update(self.asset_index.get(asset_id, set()))
        return result
    
    def filter_by_asset_class(self, asset_classes: List[str]) -> Set[str]:
        """Filter by asset classes"""
        result = set()
        for asset_class in asset_classes:
            result.update(self.asset_class_index.get(asset_class, set()))
        return result
    
    def combined_filter(self,
                        trend: Optional[int] = None,
                        volatility: Optional[int] = None,
                        structural: Optional[int] = None,
                        asset_ids: Optional[List[str]] = None,
                        asset_classes: Optional[List[str]] = None,
                        allow_adjacent: bool = True) -> Set[str]:
        """
        Apply multiple filters with intersection
        
        Returns:
            Set of trajectory IDs matching ALL criteria
        """
        result_sets = []
        
        # Regime filter
        if any(x is not None for x in [trend, volatility, structural]):
            regime_set = self.filter_by_regime(
                trend, volatility, structural, allow_adjacent
            )
            result_sets.append(regime_set)
        
        # Asset filter
        if asset_ids:
            asset_set = self.filter_by_asset(asset_ids)
            result_sets.append(asset_set)
        
        # Asset class filter
        if asset_classes:
            class_set = self.filter_by_asset_class(asset_classes)
            result_sets.append(class_set)
        
        # Return intersection
        if not result_sets:
            return set(self.trajectory_regimes.keys())
        
        result = result_sets[0]
        for s in result_sets[1:]:
            result = result.intersection(s)
        
        return result
    
    def get_regime_match_score(self,
                                query_regime: Tuple[int, int, int],
                                candidate_id: str) -> float:
        """
        Compute regime match score (0-1)
        
        Args:
            query_regime: (trend, vol, struct) of query
            candidate_id: Trajectory ID to compare
            
        Returns:
            Score from 0 (no match) to 1 (exact match)
        """
        if candidate_id not in self.trajectory_regimes:
            return 0.0
        
        candidate_regime = self.trajectory_regimes[candidate_id]
        
        # Score per dimension: 1.0 (exact), 0.5 (adjacent), 0.0 (far)
        scores = []
        for q, c in zip(query_regime, candidate_regime):
            diff = abs(q - c)
            if diff == 0:
                scores.append(1.0)
            elif diff == 1:
                scores.append(0.5)
            else:
                scores.append(0.0)
        
        return sum(scores) / len(scores)
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        regime_dist = {}
        for key, trajectories in self.regime_index.items():
            regime_dist[str(key)] = len(trajectories)
        
        return {
            'total_trajectories': self.total_count,
            'unique_regimes': len(self.regime_index),
            'unique_assets': len(self.asset_index),
            'unique_asset_classes': len(self.asset_class_index),
            'regime_distribution': regime_dist,
            'largest_regime': max(regime_dist.values()) if regime_dist else 0,
            'smallest_regime': min(regime_dist.values()) if regime_dist else 0
        }
    
    def save(self, path: str):
        """Save index to disk"""
        data = {
            'regime_index': dict(self.regime_index),
            'asset_index': dict(self.asset_index),
            'asset_class_index': dict(self.asset_class_index),
            'trajectory_regimes': self.trajectory_regimes,
            'total_count': self.total_count
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load index from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.regime_index = defaultdict(set, data['regime_index'])
        self.asset_index = defaultdict(set, data['asset_index'])
        self.asset_class_index = defaultdict(set, data['asset_class_index'])
        self.trajectory_regimes = data['trajectory_regimes']
        self.total_count = data['total_count']

