"""
HPVD Engine
===========

Main search engine combining sparse filtering and dense retrieval.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import time
import numpy as np

from .trajectory import Trajectory
from .sparse_index import SparseRegimeIndex
from .dense_index import DenseTrajectoryIndex, DenseIndexConfig
from .distance import HybridDistanceCalculator, DistanceConfig


@dataclass
class HPVDConfig:
    """Configuration for HPVD engine"""
    
    # Search parameters
    default_k: int = 25
    search_k_multiplier: int = 3
    min_candidates: int = 100
    
    # Distance config
    distance_config: DistanceConfig = None
    
    # Index config
    dense_index_config: DenseIndexConfig = None
    
    # Feature flags
    enable_sparse_filter: bool = True
    enable_reranking: bool = True
    
    def __post_init__(self):
        if self.distance_config is None:
            self.distance_config = DistanceConfig()
        if self.dense_index_config is None:
            self.dense_index_config = DenseIndexConfig()


@dataclass
class ForecastResult:
    """Probabilistic forecast with confidence intervals"""
    p_up: float
    p_down: float
    confidence_interval: Tuple[float, float]
    entropy: float


@dataclass
class AnalogResult:
    """Result from similarity search"""
    trajectory_id: str
    asset_id: str
    distance: float
    faiss_distance: float
    label_h1: int
    label_h5: int
    return_h1: float
    return_h5: float
    regime_match: float
    distance_components: Dict
    end_timestamp: Optional[str] = None


@dataclass
class SearchResult:
    """Complete search result with metadata"""
    analogs: List[AnalogResult]
    query_trajectory_id: str
    k_requested: int
    k_returned: int
    candidates_after_sparse: int
    candidates_after_dense: int
    latency_ms: float
    latency_breakdown: Dict[str, float]
    
    # Probabilistic forecasts
    forecast_h1: Optional[ForecastResult] = None
    forecast_h5: Optional[ForecastResult] = None
    
    # Quality metrics
    aci: float = 0.0
    regime_coherence: float = 0.0
    
    # Abstention
    should_abstain: bool = False
    abstention_reason: str = ""


class HPVDEngine:
    """
    Hybrid Probabilistic Vector Database Engine
    
    Main entry point for trajectory similarity search.
    
    Pipeline:
    1. Sparse filtering by regime
    2. Dense FAISS search
    3. Hybrid distance reranking
    4. Quality assessment
    """
    
    def __init__(self, config: HPVDConfig = None):
        self.config = config or HPVDConfig()
        
        # Indexes
        self.sparse_index: Optional[SparseRegimeIndex] = None
        self.dense_index: Optional[DenseTrajectoryIndex] = None
        
        # Trajectory storage
        self.trajectories: Dict[str, Trajectory] = {}
        
        # Distance calculator
        self.distance_calc = HybridDistanceCalculator(self.config.distance_config)
        
        # State
        self.is_built = False
    
    def build(self, trajectories: List[Trajectory]):
        """
        Build indexes from trajectory list
        
        Args:
            trajectories: List of Trajectory objects
        """
        print(f"Building HPVD with {len(trajectories)} trajectories...")
        start_time = time.time()
        
        # Validate trajectories
        valid_trajectories = [t for t in trajectories if t.validate()]
        if len(valid_trajectories) < len(trajectories):
            print(f"Warning: {len(trajectories) - len(valid_trajectories)} invalid trajectories skipped")
        
        # Store trajectories
        for traj in valid_trajectories:
            self.trajectories[traj.trajectory_id] = traj
        
        # Build sparse index
        self.sparse_index = SparseRegimeIndex()
        for traj in valid_trajectories:
            self.sparse_index.add(
                trajectory_id=traj.trajectory_id,
                trend=traj.trend_regime,
                volatility=traj.volatility_regime,
                structural=traj.structural_regime,
                asset_id=traj.asset_id,
                asset_class=traj.asset_class
            )
        
        # Build dense index
        embeddings = np.array([t.embedding for t in valid_trajectories])
        trajectory_ids = [t.trajectory_id for t in valid_trajectories]
        
        self.dense_index = DenseTrajectoryIndex(self.config.dense_index_config)
        self.dense_index.build(embeddings, trajectory_ids)
        
        self.is_built = True
        
        elapsed = time.time() - start_time
        print(f"HPVD built in {elapsed:.2f}s")
        print(f"  Sparse index: {self.sparse_index.get_statistics()['unique_regimes']} regimes")
        print(f"  Dense index: {self.dense_index.ntotal} vectors")
    
    def search(self,
               query_trajectory: Trajectory,
               k: int = None) -> SearchResult:
        """
        Find k most similar trajectories
        
        Args:
            query_trajectory: Query trajectory
            k: Number of results (default: config.default_k)
            
        Returns:
            SearchResult with analogs and metadata
        """
        if not self.is_built:
            raise RuntimeError("HPVD not built. Call build() first.")
        
        k = k or self.config.default_k
        latency = {}
        
        total_start = time.time()
        
        # ========== STAGE 1: Sparse Filtering ==========
        stage_start = time.time()
        
        if self.config.enable_sparse_filter:
            candidate_ids = self.sparse_index.combined_filter(
                trend=query_trajectory.trend_regime,
                volatility=query_trajectory.volatility_regime,
                structural=query_trajectory.structural_regime,
                allow_adjacent=True
            )
            
            # Fallback if too few candidates
            if len(candidate_ids) < self.config.min_candidates:
                candidate_ids = self.sparse_index.combined_filter(
                    trend=query_trajectory.trend_regime,
                    allow_adjacent=True
                )
            
            if len(candidate_ids) < k:
                candidate_ids = set(self.trajectories.keys())
        else:
            candidate_ids = set(self.trajectories.keys())
        
        candidates_after_sparse = len(candidate_ids)
        latency['sparse_filter_ms'] = (time.time() - stage_start) * 1000
        
        # ========== STAGE 2: Dense Retrieval ==========
        stage_start = time.time()
        
        search_k = k * self.config.search_k_multiplier
        dense_results = self.dense_index.search_with_filter(
            query_embedding=query_trajectory.embedding,
            candidate_ids=candidate_ids,
            k=search_k
        )
        
        candidates_after_dense = len(dense_results)
        latency['dense_search_ms'] = (time.time() - stage_start) * 1000
        
        # ========== STAGE 3: Hybrid Reranking ==========
        stage_start = time.time()
        
        if self.config.enable_reranking:
            reranked = []
            query_regime = query_trajectory.get_regime_tuple()
            
            for tid, faiss_dist in dense_results:
                traj = self.trajectories.get(tid)
                if traj is None:
                    continue
                
                hybrid_dist, components = self.distance_calc.compute(
                    query_trajectory.matrix,
                    traj.matrix,
                    query_regime,
                    traj.get_regime_tuple()
                )
                
                reranked.append({
                    'trajectory_id': tid,
                    'asset_id': traj.asset_id,
                    'hybrid_distance': hybrid_dist,
                    'faiss_distance': faiss_dist,
                    'label_h1': traj.label_h1,
                    'label_h5': traj.label_h5,
                    'return_h1': traj.return_h1,
                    'return_h5': traj.return_h5,
                    'regime_match': components['regime_match'],
                    'components': components,
                    'end_timestamp': traj.end_timestamp.isoformat()
                })
            
            reranked.sort(key=lambda x: x['hybrid_distance'])
        else:
            reranked = []
            for tid, faiss_dist in dense_results:
                traj = self.trajectories.get(tid)
                if traj is None:
                    continue
                
                regime_match = self.sparse_index.get_regime_match_score(
                    query_trajectory.get_regime_tuple(), tid
                )
                
                reranked.append({
                    'trajectory_id': tid,
                    'asset_id': traj.asset_id,
                    'hybrid_distance': faiss_dist,
                    'faiss_distance': faiss_dist,
                    'label_h1': traj.label_h1,
                    'label_h5': traj.label_h5,
                    'return_h1': traj.return_h1,
                    'return_h5': traj.return_h5,
                    'regime_match': regime_match,
                    'components': {},
                    'end_timestamp': traj.end_timestamp.isoformat()
                })
        
        latency['reranking_ms'] = (time.time() - stage_start) * 1000
        
        # ========== STAGE 4: Format Results ==========
        analogs = [
            AnalogResult(
                trajectory_id=r['trajectory_id'],
                asset_id=r['asset_id'],
                distance=r['hybrid_distance'],
                faiss_distance=r['faiss_distance'],
                label_h1=r['label_h1'],
                label_h5=r['label_h5'],
                return_h1=r['return_h1'],
                return_h5=r['return_h5'],
                regime_match=r['regime_match'],
                distance_components=r['components'],
                end_timestamp=r['end_timestamp']
            )
            for r in reranked[:k]
        ]
        
        # ========== STAGE 5: Quality Assessment ==========
        stage_start = time.time()
        
        aci = self._compute_aci(analogs)
        regime_coherence = self._compute_regime_coherence(analogs)
        forecast_h1 = self._compute_forecast(analogs, 'h1')
        forecast_h5 = self._compute_forecast(analogs, 'h5')
        
        # Check abstention
        should_abstain = False
        abstention_reason = ""
        
        if forecast_h1.entropy > 0.9:
            should_abstain = True
            abstention_reason = f"High H1 entropy: {forecast_h1.entropy:.3f}"
        elif aci < 0.7:
            should_abstain = True
            abstention_reason = f"Low ACI: {aci:.3f}"
        
        latency['quality_ms'] = (time.time() - stage_start) * 1000
        latency['total_ms'] = (time.time() - total_start) * 1000
        
        return SearchResult(
            analogs=analogs,
            query_trajectory_id=query_trajectory.trajectory_id,
            k_requested=k,
            k_returned=len(analogs),
            candidates_after_sparse=candidates_after_sparse,
            candidates_after_dense=candidates_after_dense,
            latency_ms=latency['total_ms'],
            latency_breakdown=latency,
            forecast_h1=forecast_h1,
            forecast_h5=forecast_h5,
            aci=aci,
            regime_coherence=regime_coherence,
            should_abstain=should_abstain,
            abstention_reason=abstention_reason
        )
    
    def _compute_aci(self, analogs: List[AnalogResult]) -> float:
        """Compute Analog Cohesion Index"""
        if len(analogs) < 2:
            return 1.0
        
        distances = np.array([a.distance for a in analogs])
        mean_dist = distances.mean()
        std_dist = distances.std()
        
        aci = 1.0 - (mean_dist + std_dist) / 2.0
        return float(np.clip(aci, 0.0, 1.0))
    
    def _compute_regime_coherence(self, analogs: List[AnalogResult]) -> float:
        """Compute regime coherence"""
        if not analogs:
            return 0.0
        
        scores = [a.regime_match for a in analogs]
        return float(np.mean(scores))
    
    def _compute_forecast(self, analogs: List[AnalogResult], horizon: str) -> ForecastResult:
        """Compute probabilistic forecast"""
        if not analogs:
            return ForecastResult(p_up=0.5, p_down=0.5, 
                                   confidence_interval=(0.0, 1.0), entropy=1.0)
        
        # Distance-based weights
        alpha = 2.0
        distances = np.array([a.distance for a in analogs])
        weights = np.exp(-alpha * distances)
        weights = weights / weights.sum()
        
        # Get outcomes
        if horizon == 'h1':
            outcomes = np.array([1.0 if a.label_h1 == 1 else 0.0 for a in analogs])
        else:
            outcomes = np.array([1.0 if a.label_h5 == 1 else 0.0 for a in analogs])
        
        # Weighted probability
        p_up = float(np.dot(weights, outcomes))
        p_down = 1.0 - p_up
        
        # Wilson score interval
        n = len(analogs)
        z = 1.96
        
        denominator = 1 + z**2 / n
        center = (p_up + z**2 / (2*n)) / denominator
        spread = z * np.sqrt((p_up * p_down) / n + z**2 / (4*n**2)) / denominator
        
        ci_lower = max(0.0, center - spread)
        ci_upper = min(1.0, center + spread)
        
        # Entropy
        if p_up <= 0 or p_up >= 1:
            entropy = 0.0
        else:
            entropy = -p_up * np.log2(p_up) - p_down * np.log2(p_down)
        
        return ForecastResult(
            p_up=p_up,
            p_down=p_down,
            confidence_interval=(ci_lower, ci_upper),
            entropy=float(entropy)
        )
    
    def get_statistics(self) -> Dict:
        """Get HPVD statistics"""
        return {
            'total_trajectories': len(self.trajectories),
            'sparse_index_stats': self.sparse_index.get_statistics() if self.sparse_index else {},
            'dense_index_vectors': self.dense_index.ntotal if self.dense_index else 0,
            'is_built': self.is_built
        }
    
    def save(self, path: str):
        """Save HPVD to disk"""
        import os
        import pickle
        
        os.makedirs(path, exist_ok=True)
        
        self.sparse_index.save(f"{path}/sparse_index.pkl")
        self.dense_index.save(f"{path}/dense_index")
        
        with open(f"{path}/trajectories.pkl", 'wb') as f:
            pickle.dump(self.trajectories, f)
        
        with open(f"{path}/config.pkl", 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"HPVD saved to {path}")
    
    def load(self, path: str):
        """Load HPVD from disk"""
        import pickle
        
        with open(f"{path}/config.pkl", 'rb') as f:
            self.config = pickle.load(f)
        
        with open(f"{path}/trajectories.pkl", 'rb') as f:
            self.trajectories = pickle.load(f)
        
        self.sparse_index = SparseRegimeIndex()
        self.sparse_index.load(f"{path}/sparse_index.pkl")
        
        self.dense_index = DenseTrajectoryIndex(self.config.dense_index_config)
        self.dense_index.load(f"{path}/dense_index")
        
        self.distance_calc = HybridDistanceCalculator(self.config.distance_config)
        
        self.is_built = True
        print(f"HPVD loaded from {path}: {len(self.trajectories)} trajectories")

