"""
HPVD Engine
===========

Main search engine combining sparse filtering and dense retrieval.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Union
from datetime import datetime
import json
import time
import uuid
import warnings
import numpy as np

from .trajectory import Trajectory, HPVDInputBundle
from .sparse_index import SparseRegimeIndex
from .dense_index import DenseTrajectoryIndex, DenseIndexConfig
from .distance import HybridDistanceCalculator, DistanceConfig
from .embedding import EmbeddingComputer
from .dna_similarity import DNASimilarityCalculator, DNASimilarityConfig
from .family import (
    FamilyFormationEngine,
    FamilyFormationConfig,
    AnalogFamily,
    FamilyMember,
    FamilyCoherence,
    StructuralSignature,
    UncertaintyFlags,
)


@dataclass
class HPVDConfig:
    """Configuration for HPVD engine"""
    
    # Search parameters
    default_k: int = 25
    search_k_multiplier: int = 3
    min_candidates: int = 100
    
    # Embedding
    embedding_dim: int = 256
    
    # Multi-channel fusion: weight of DNA similarity in fused distance
    # fused = (1 - dna_weight) * trajectory_dist + dna_weight * dna_dist
    dna_similarity_weight: float = 0.3
    
    # Distance config
    distance_config: DistanceConfig = None
    
    # Index config
    dense_index_config: DenseIndexConfig = None
    
    # Family formation config
    family_config: FamilyFormationConfig = None
    
    # Feature flags
    enable_sparse_filter: bool = True
    enable_reranking: bool = True
    
    def __post_init__(self):
        if self.distance_config is None:
            self.distance_config = DistanceConfig()
        if self.dense_index_config is None:
            self.dense_index_config = DenseIndexConfig()
        if self.family_config is None:
            self.family_config = FamilyFormationConfig()


@dataclass
class ForecastResult:
    """
    Placeholder container for downstream probabilistic reasoning.

    Matrix22 note:
        - HPVD core must **not** compute probabilities, entropy, or
          confidence intervals based on outcomes.
        - These fields are kept only for backward-compat with existing
          demos and will be populated in a structurally neutral way.
        - Real probabilistic logic belongs in PMR-DB / adapters, not HPVD.
    """

    p_up: float
    p_down: float
    confidence_interval: Tuple[float, float]
    entropy: float


# Note: FamilyMember, FamilyCoherence, StructuralSignature, UncertaintyFlags,
# and AnalogFamily are now imported from .family module


@dataclass
class HPVD_Output:
    """
    HPVD Output - structured empirical evidence in the form of Analog Families.
    
    Matrix22: HPVD outputs structured empirical evidence, NOT predictions/probabilities/decisions.
    """
    analog_families: List[AnalogFamily]
    retrieval_diagnostics: Dict[str, int]  # e.g., candidates_considered, families_formed, rejected_candidates
    metadata: Dict[str, str]  # hpvd_version, query_id, schema_version, timestamp

    # ------------------------------------------------------------------
    # Serialization helpers (hpvd_output_v1 contract)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Serialize to a plain dict following the ``hpvd_output_v1`` schema.

        The output is JSON-safe (no numpy/datetime objects).
        """
        return {
            "metadata": dict(self.metadata),
            "retrieval_diagnostics": {
                k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                for k, v in self.retrieval_diagnostics.items()
            },
            "analog_families": [
                {
                    "family_id": af.family_id,
                    "members": [
                        {
                            "trajectory_id": m.trajectory_id,
                            "confidence": float(m.confidence),
                        }
                        for m in af.members
                    ],
                    "coherence": {
                        "mean_confidence": float(af.coherence.mean_confidence),
                        "dispersion": float(af.coherence.dispersion),
                        "size": int(af.coherence.size),
                    },
                    "structural_signature": {
                        "phase": af.structural_signature.phase,
                        "avg_K": (
                            float(af.structural_signature.avg_K)
                            if af.structural_signature.avg_K is not None
                            else None
                        ),
                        "avg_LTV": (
                            float(af.structural_signature.avg_LTV)
                            if af.structural_signature.avg_LTV is not None
                            else None
                        ),
                        "avg_LVC": (
                            float(af.structural_signature.avg_LVC)
                            if af.structural_signature.avg_LVC is not None
                            else None
                        ),
                    },
                    "uncertainty_flags": {
                        "phase_boundary": af.uncertainty_flags.phase_boundary,
                        "weak_support": af.uncertainty_flags.weak_support,
                        "partial_overlap": af.uncertainty_flags.partial_overlap,
                    },
                }
                for af in self.analog_families
            ],
        }

    def to_json(self, **kwargs) -> str:
        """Return a JSON string (``hpvd_output_v1`` schema).

        Extra *kwargs* are forwarded to ``json.dumps`` (e.g. ``indent=2``).
        """
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict) -> "HPVD_Output":
        """Reconstruct an ``HPVD_Output`` from a dict produced by ``to_dict()``.

        Raises ``ValueError`` if required keys are missing or
        ``schema_version`` is not ``hpvd_output_v1``.
        """
        # --- validate top-level keys ---
        for key in ("metadata", "retrieval_diagnostics", "analog_families"):
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        schema = data["metadata"].get("schema_version")
        if schema != "hpvd_output_v1":
            raise ValueError(
                f"Unsupported schema_version: {schema!r} (expected 'hpvd_output_v1')"
            )

        families: List[AnalogFamily] = []
        for af_data in data["analog_families"]:
            members = [
                FamilyMember(
                    trajectory_id=m["trajectory_id"],
                    confidence=float(m["confidence"]),
                )
                for m in af_data["members"]
            ]
            coherence = FamilyCoherence(
                mean_confidence=float(af_data["coherence"]["mean_confidence"]),
                dispersion=float(af_data["coherence"]["dispersion"]),
                size=int(af_data["coherence"]["size"]),
            )
            sig = af_data["structural_signature"]
            structural_signature = StructuralSignature(
                phase=sig["phase"],
                avg_K=float(sig["avg_K"]) if sig.get("avg_K") is not None else None,
                avg_LTV=float(sig["avg_LTV"]) if sig.get("avg_LTV") is not None else None,
                avg_LVC=float(sig["avg_LVC"]) if sig.get("avg_LVC") is not None else None,
            )
            uf = af_data["uncertainty_flags"]
            uncertainty_flags = UncertaintyFlags(
                phase_boundary=bool(uf["phase_boundary"]),
                weak_support=bool(uf["weak_support"]),
                partial_overlap=bool(uf["partial_overlap"]),
            )
            families.append(
                AnalogFamily(
                    family_id=af_data["family_id"],
                    members=members,
                    coherence=coherence,
                    structural_signature=structural_signature,
                    uncertainty_flags=uncertainty_flags,
                )
            )

        return cls(
            analog_families=families,
            retrieval_diagnostics=dict(data["retrieval_diagnostics"]),
            metadata=dict(data["metadata"]),
        )


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
    
    # Probabilistic forecasts (Matrix22: structural placeholders only)
    forecast_h1: Optional[ForecastResult] = None
    forecast_h5: Optional[ForecastResult] = None
    
    # Quality metrics
    aci: float = 0.0
    regime_coherence: float = 0.0
    
    # Abstention (Matrix22: decision must live in Authorization / PMR)
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
        
        # Embedding computer (PCA-based)
        self.embedding_computer = EmbeddingComputer(self.config.embedding_dim)
        
        # DNA similarity calculator
        self.dna_calculator = DNASimilarityCalculator()
        
        # Family formation engine
        self.family_engine = FamilyFormationEngine(self.config.family_config)
        
        # State
        self.is_built = False
    
    def build(self, trajectories: List[Trajectory]):
        """
        Build indexes from trajectory list.

        .. deprecated::
            Use ``build_from_bundles(bundles)`` instead.
            This method is kept for backward compatibility only.

        Args:
            trajectories: List of Trajectory objects
        """
        warnings.warn(
            "HPVDEngine.build() is deprecated. "
            "Use build_from_bundles(List[HPVDInputBundle]) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
    
    @staticmethod
    def _extract_regime_from_bundle(bundle: HPVDInputBundle) -> Tuple[int, int, int]:
        """Extract regime tuple from bundle metadata."""
        trend, vol, struct = 0, 0, 0
        if 'regime_id' in bundle.metadata:
            regime_id = bundle.metadata['regime_id']
            if 'R1' in regime_id:
                trend, vol, struct = 1, 0, 1
            elif 'R2' in regime_id:
                trend, vol, struct = -1, 0, -1
            elif 'R3' in regime_id:
                trend, vol, struct = 0, 1, 1
            elif 'R5' in regime_id:
                trend, vol, struct = 1, 1, -1
        return trend, vol, struct

    def _bundle_to_trajectory(
        self,
        bundle: HPVDInputBundle,
        embedding: Optional[np.ndarray] = None,
    ) -> Trajectory:
        """
        Convert HPVDInputBundle to Trajectory for internal use.

        Args:
            bundle: Input bundle with trajectory, dna, geometry, metadata.
            embedding: Pre-computed PCA embedding.  When *None* the engine
                       falls back to a naive truncation (for queries when
                       PCA is already fitted, use ``self.embedding_computer``
                       from the caller).
        """
        if embedding is None:
            if self.embedding_computer.is_fitted:
                embedding = self.embedding_computer.transform(bundle.trajectory)
            else:
                # Fallback: naive truncation (pre-PCA build path)
                flat_matrix = bundle.trajectory.flatten()
                if len(flat_matrix) > self.config.embedding_dim:
                    embedding = flat_matrix[: self.config.embedding_dim].astype(np.float32)
                else:
                    embedding = np.pad(
                        flat_matrix,
                        (0, self.config.embedding_dim - len(flat_matrix)),
                        mode="constant",
                    ).astype(np.float32)

        trend, vol, struct = self._extract_regime_from_bundle(bundle)

        return Trajectory(
            trajectory_id=bundle.metadata.get('trajectory_id', str(uuid.uuid4())),
            asset_id=bundle.metadata.get('asset_id', 'synthetic'),
            end_timestamp=(
                datetime.fromisoformat(bundle.metadata['timestamp'])
                if 'timestamp' in bundle.metadata
                else datetime.now()
            ),
            matrix=bundle.trajectory,
            embedding=embedding,
            dna=bundle.dna.astype(np.float32),
            trend_regime=trend,
            volatility_regime=vol,
            structural_regime=struct,
            asset_class='synthetic',
        )
    
    def build_from_bundles(self, bundles: List[HPVDInputBundle]):
        """
        Build HPVD from HPVDInputBundle list (Matrix22 canonical method).

        Steps:
            1. Fit PCA embedding on all trajectory matrices.
            2. Batch-transform matrices into meaningful 256-dim embeddings.
            3. Convert bundles to Trajectory objects (with PCA embeddings + DNA).
            4. Delegate to ``self.build()`` for index construction.

        Args:
            bundles: List of HPVDInputBundle objects
        """
        if not bundles:
            self.build([])
            return

        # Collect all matrices and fit PCA
        matrices = np.array([b.trajectory for b in bundles])
        self.embedding_computer.fit(matrices)

        # Batch-transform to get meaningful embeddings
        embeddings = self.embedding_computer.transform_batch(matrices)

        # Convert each bundle with its pre-computed embedding
        trajectories = [
            self._bundle_to_trajectory(bundle, embedding=embeddings[i])
            for i, bundle in enumerate(bundles)
        ]

        self.build(trajectories)
    
    def search_families(self,
                        query: Union[HPVDInputBundle, Trajectory],
                        max_candidates: int = None) -> HPVD_Output:
        """
        Matrix22: Find analog families (new canonical method).
        
        Returns structured empirical evidence as Analog Families, NOT predictions.
        
        Args:
            query: Query as HPVDInputBundle (preferred) or Trajectory (legacy)
            max_candidates: Maximum candidates to consider (default: config.default_k * 5)
            
        Returns:
            HPVD_Output with analog families, diagnostics, and metadata
        """
        if not self.is_built:
            raise RuntimeError("HPVD not built. Call build() first.")
        
        # Convert HPVDInputBundle to Trajectory if needed
        if isinstance(query, HPVDInputBundle):
            query.validate()
        elif isinstance(query, Trajectory):
            warnings.warn(
                "Passing Trajectory to search_families() is deprecated. "
                "Use HPVDInputBundle instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if isinstance(query, HPVDInputBundle):
            query_trajectory = self._bundle_to_trajectory(query)
        else:
            query_trajectory = query
        
        max_candidates = max_candidates or (self.config.default_k * 5)
        latency = {}
        total_start = time.time()
        
        # ========== STAGE 1: Evolutionary Compatibility Screening ==========
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
            
            if len(candidate_ids) < 10:
                candidate_ids = set(self.trajectories.keys())
        else:
            candidate_ids = set(self.trajectories.keys())
        
        candidates_after_screening = len(candidate_ids)
        latency['screening_ms'] = (time.time() - stage_start) * 1000
        
        # ========== STAGE 2: Candidate Retrieval (broad, permissive) ==========
        stage_start = time.time()
        
        search_k = min(max_candidates, len(candidate_ids))
        dense_results = self.dense_index.search_with_filter(
            query_embedding=query_trajectory.embedding,
            candidate_ids=candidate_ids,
            k=search_k
        )
        
        candidates_retrieved = len(dense_results)
        latency['retrieval_ms'] = (time.time() - stage_start) * 1000
        
        # ========== STAGE 3: Multi-Channel Similarity Evaluation ==========
        stage_start = time.time()
        
        query_regime = query_trajectory.get_regime_tuple()
        dna_weight = self.config.dna_similarity_weight
        traj_weight = 1.0 - dna_weight
        evaluated_candidates = []
        
        for tid, faiss_dist in dense_results:
            traj = self.trajectories.get(tid)
            if traj is None:
                continue
            
            # Channel 1: Trajectory hybrid distance (euclidean + cosine + temporal + regime)
            hybrid_dist, components = self.distance_calc.compute(
                query_trajectory.matrix,
                traj.matrix,
                query_regime,
                traj.get_regime_tuple()
            )
            
            # Channel 2: DNA similarity (evolutionary phase compatibility)
            dna_sim, dna_components = self.dna_calculator.compute(
                query_trajectory.dna,
                traj.dna,
            )
            dna_distance = 1.0 - dna_sim
            
            # Multi-channel fusion
            fused_distance = traj_weight * hybrid_dist + dna_weight * dna_distance
            
            # Confidence = inverse of normalized fused distance
            # Matrix22: This is descriptive, not a probability
            confidence = max(0.0, 1.0 - min(fused_distance, 1.0))
            
            evaluated_candidates.append({
                'trajectory_id': tid,
                'trajectory': traj,
                'confidence': confidence,
                'hybrid_distance': hybrid_dist,
                'fused_distance': fused_distance,
                'dna_similarity': dna_sim,
                'dna_components': dna_components,
                'regime_match': components['regime_match'],
                'distance_components': components,
                'regime_tuple': traj.get_regime_tuple()
            })
        
        latency['evaluation_ms'] = (time.time() - stage_start) * 1000
        
        # ========== STAGE 4: Probabilistic Neighborhood Admission ==========
        # Matrix22: Simple threshold-based admission for now
        # TODO: Implement proper probabilistic admission
        stage_start = time.time()
        
        admission_threshold = 0.3  # Minimum confidence to be admitted
        admitted = [
            c for c in evaluated_candidates
            if c['confidence'] >= admission_threshold
        ]
        rejected = [
            c for c in evaluated_candidates
            if c['confidence'] < admission_threshold
        ]
        
        latency['admission_ms'] = (time.time() - stage_start) * 1000
        
        # ========== STAGE 5: Analog Family Formation ==========
        stage_start = time.time()
        
        families = self._form_analog_families(admitted, query_regime)
        
        latency['family_formation_ms'] = (time.time() - stage_start) * 1000
        
        # ========== STAGE 6: Uncertainty Annotation ==========
        # (Already computed during family formation)
        
        # ========== STAGE 7: Output Assembly ==========
        latency['total_ms'] = (time.time() - total_start) * 1000
        
        diagnostics = {
            'candidates_considered': candidates_after_screening,
            'candidates_retrieved': candidates_retrieved,
            'candidates_admitted': len(admitted),
            'candidates_rejected': len(rejected),
            'families_formed': len(families),
            'latency_ms': latency['total_ms']
        }
        
        # Extract metadata from query (prefer bundle metadata if available)
        if isinstance(query, HPVDInputBundle):
            query_id = query.metadata.get('trajectory_id', query_trajectory.trajectory_id)
            query_timestamp = query.metadata.get('timestamp', query_trajectory.end_timestamp.isoformat())
        else:
            query_id = query_trajectory.trajectory_id
            query_timestamp = query_trajectory.end_timestamp.isoformat() if hasattr(query_trajectory.end_timestamp, 'isoformat') else str(query_trajectory.end_timestamp)
        
        metadata = {
            'hpvd_version': 'v1',
            'query_id': query_id,
            'schema_version': 'hpvd_output_v1',
            'timestamp': query_timestamp
        }
        
        return HPVD_Output(
            analog_families=families,
            retrieval_diagnostics=diagnostics,
            metadata=metadata
        )
    
    def _form_analog_families(self,
                               admitted_candidates: List[Dict],
                               query_regime: Tuple[int, int, int]) -> List[AnalogFamily]:
        """
        Form analog families from admitted candidates.
        
        Delegates to FamilyFormationEngine for actual family formation logic.
        
        Matrix22: Families are not forced to be compact, may overlap,
        and may be small or large. HPVD never forces a single family or
        merges incompatible groups.
        """
        return self.family_engine.form_families(admitted_candidates, query_regime)
    
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
        
        # ========== STAGE 3: Hybrid Reranking (with DNA fusion) ==========
        stage_start = time.time()
        
        dna_w = self.config.dna_similarity_weight
        traj_w = 1.0 - dna_w
        
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
                
                # DNA channel
                dna_sim, _ = self.dna_calculator.compute(
                    query_trajectory.dna, traj.dna
                )
                fused_distance = traj_w * hybrid_dist + dna_w * (1.0 - dna_sim)
                
                reranked.append({
                    'trajectory_id': tid,
                    'asset_id': traj.asset_id,
                    'hybrid_distance': fused_distance,
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
        
        # ========== STAGE 5: Quality Assessment (descriptive only) ==========
        stage_start = time.time()
        
        aci = self._compute_aci(analogs)
        regime_coherence = self._compute_regime_coherence(analogs)
        forecast_h1 = self._compute_forecast(analogs, 'h1')
        forecast_h5 = self._compute_forecast(analogs, 'h5')
        
        # Matrix22: HPVD core must not decide abstention.
        # Abstention is delegated to downstream authorization / PMR layers.
        should_abstain = False
        abstention_reason = "abstention delegated to PMR/authorization layer"
        
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
        """
        Compute structurally neutral placeholder for probabilistic forecast.

        Matrix22:
            - This method MUST NOT use labels/returns or any future outcomes.
            - It only reports a symmetric, maximally-uncertain prior that
              downstream systems may later update.
        """
        if not analogs:
            return ForecastResult(
                p_up=0.5,
                p_down=0.5,
                confidence_interval=(0.0, 1.0),
                entropy=1.0,
            )

        # Always return neutral, high-entropy prior regardless of analog labels.
        # This keeps HPVD outcome-blind while preserving the existing schema.
        return ForecastResult(
            p_up=0.5,
            p_down=0.5,
            confidence_interval=(0.0, 1.0),
            entropy=1.0,
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
        
        if self.embedding_computer.is_fitted:
            self.embedding_computer.save(f"{path}/embedding_pca.pkl")
        
        with open(f"{path}/trajectories.pkl", 'wb') as f:
            pickle.dump(self.trajectories, f)
        
        with open(f"{path}/config.pkl", 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"HPVD saved to {path}")
    
    def load(self, path: str):
        """Load HPVD from disk"""
        import os
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
        
        # Restore embedding computer if saved
        pca_path = f"{path}/embedding_pca.pkl"
        self.embedding_computer = EmbeddingComputer(self.config.embedding_dim)
        if os.path.exists(pca_path):
            self.embedding_computer.load(pca_path)
        
        # Restore DNA calculator
        self.dna_calculator = DNASimilarityCalculator()
        
        self.is_built = True
        print(f"HPVD loaded from {path}: {len(self.trajectories)} trajectories")

