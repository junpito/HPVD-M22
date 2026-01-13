"""
Synthetic Data Generator for HPVD Testing
==========================================

Generates outcome-blind synthetic data for HPVD according to
Separated Development Guide Section 1 & 5.

Key principles:
- NO outcomes, NO labels, NO returns
- Deterministic (seed-based) for replayability
- Tests specific epistemic capabilities
- Produces HPVDInputBundle (not Trajectory with outcomes)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .trajectory import HPVDInputBundle


@dataclass
class RegimeDefinition:
    """Definition of a structural regime for synthetic generation"""
    regime_id: str  # R1, R2, R3, etc.
    description: str
    trajectory_behavior: str  # How trajectories evolve
    dna_signature: np.ndarray  # Phase encoding (K-dim)
    curvature_variance: float  # For trajectory generation
    smoothness: float  # 0-1, higher = smoother


class SyntheticDataGenerator:
    """
    Generate synthetic HPVD input bundles for testing.
    
    Implements 5 canonical scenarios from Separated Development Guide:
    - Scenario A: Clean Regime Repetition
    - Scenario B: Surface Similarity, Different Evolution
    - Scenario C: Same Phase, Different Amplitude
    - Scenario D: Transitional Phase (Ambiguous)
    - Scenario E: Novel Structure (No Analogs)
    """
    
    def __init__(self, 
                 trajectory_horizon: int = 60,
                 state_dim: int = 45,
                 dna_dim: int = 16,
                 seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            trajectory_horizon: T (time steps), default 60
            state_dim: D (state dimensions), default 45
            dna_dim: K (DNA dimensions), default 16
            seed: Random seed for reproducibility
        """
        self.T = trajectory_horizon
        self.D = state_dim
        self.K = dna_dim
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        
        # Base date for synthetic timestamps (ISO format)
        self.base_date = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # Define 6 regimes from guide
        self.regimes = self._define_regimes()
    
    def _define_regimes(self) -> Dict[str, RegimeDefinition]:
        """Define structural regimes (R1-R6) from guide"""
        regimes = {}
        
        # R1: Stable expansion
        regimes['R1'] = RegimeDefinition(
            regime_id='R1',
            description='Stable expansion (smooth, low curvature)',
            trajectory_behavior='Smooth drift along stable manifold',
            dna_signature=self._create_dna_signature([1.0, 0.8, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0]),
            curvature_variance=0.1,
            smoothness=0.9
        )
        
        # R2: Stable contraction
        regimes['R2'] = RegimeDefinition(
            regime_id='R2',
            description='Stable contraction',
            trajectory_behavior='Smooth convergence',
            dna_signature=self._create_dna_signature([-1.0, 0.8, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0]),
            curvature_variance=0.1,
            smoothness=0.9
        )
        
        # R3: Compression / crowding
        regimes['R3'] = RegimeDefinition(
            regime_id='R3',
            description='Compression / crowding',
            trajectory_behavior='Converging states, reduced dispersion',
            dna_signature=self._create_dna_signature([0.0, -0.5, 0.9, 0.3, 0.1, 0.0, 0.0, 0.0]),
            curvature_variance=0.3,
            smoothness=0.6
        )
        
        # R4: Transitional (phase boundary)
        regimes['R4'] = RegimeDefinition(
            regime_id='R4',
            description='Transitional (phase boundary)',
            trajectory_behavior='Directional ambiguity, oscillation',
            dna_signature=self._create_dna_signature([0.0, 0.0, 0.0, 0.5, 0.5, 0.3, 0.2, 0.1]),
            curvature_variance=0.5,
            smoothness=0.4
        )
        
        # R5: Structural stress
        regimes['R5'] = RegimeDefinition(
            regime_id='R5',
            description='Structural stress (high curvature variability)',
            trajectory_behavior='Rapid directional changes',
            dna_signature=self._create_dna_signature([0.2, -0.3, -0.2, 0.8, -0.5, 0.4, -0.3, 0.2]),
            curvature_variance=0.8,
            smoothness=0.2
        )
        
        # R6: Novel / unseen
        regimes['R6'] = RegimeDefinition(
            regime_id='R6',
            description='Novel / unseen regime',
            trajectory_behavior='Unseen geometry, undefined phase',
            dna_signature=self._create_dna_signature([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Sparse/undefined
            curvature_variance=1.0,
            smoothness=0.1
        )
        
        return regimes
    
    def _create_dna_signature(self, values: List[float]) -> np.ndarray:
        """Create DNA signature vector, padding to K dimensions"""
        dna = np.array(values, dtype=np.float32)
        if len(dna) < self.K:
            # Pad with zeros
            dna = np.pad(dna, (0, self.K - len(dna)), mode='constant')
        elif len(dna) > self.K:
            dna = dna[:self.K]
        return dna.astype(np.float32)
    
    def _generate_trajectory_from_regime(self,
                                        regime: RegimeDefinition,
                                        noise_level: float = 0.1,
                                        scale: float = 1.0) -> np.ndarray:
        """
        Generate trajectory matrix (T × D) from regime definition.
        
        Args:
            regime: Regime definition
            noise_level: Stochastic perturbation level (0-1)
            scale: Amplitude scaling factor
            
        Returns:
            (T, D) trajectory matrix
        """
        # Base latent path (low-dimensional)
        latent_dim = min(5, self.D // 3)
        latent_path = self.rng.randn(self.T, latent_dim).astype(np.float32)
        
        # Apply regime-specific evolution
        if regime.smoothness > 0.5:
            # Smooth evolution: apply moving average
            for t in range(1, self.T):
                latent_path[t] = (1 - regime.smoothness) * latent_path[t] + \
                                 regime.smoothness * latent_path[t-1]
        
        # Embed into D dimensions
        embedding_matrix = self.rng.randn(latent_dim, self.D).astype(np.float32)
        trajectory = latent_path @ embedding_matrix
        
        # Apply regime-specific transformations
        # Curvature changes
        curvature_weights = np.linspace(0, 1, self.T) ** regime.curvature_variance
        trajectory = trajectory * curvature_weights.reshape(-1, 1)
        
        # Scale
        trajectory = trajectory * scale
        
        # Add controlled noise
        noise = self.rng.randn(self.T, self.D).astype(np.float32) * noise_level
        trajectory = trajectory + noise
        
        return trajectory.astype(np.float32)
    
    def generate_scenario_a(self,
                           n_historical: int = 20,
                           n_query: int = 1,
                           regime_id: str = 'R1',
                           noise_level: float = 0.1) -> Dict[str, List[HPVDInputBundle]]:
        """
        Scenario A: Clean Regime Repetition
        
        Generate multiple trajectories with:
        - Same structural evolution
        - Same phase (DNA)
        - Small noise differences
        - Different "historical times"
        
        Expected HPVD behavior:
        - Retrieves all correct analogs
        - Groups into ONE analog family
        - High confidence, low dispersion
        """
        regime = self.regimes[regime_id]
        
        historical = []
        for i in range(n_historical):
            # Same regime, different noise realization
            trajectory = self._generate_trajectory_from_regime(
                regime, noise_level=noise_level + self.rng.uniform(0, 0.05)
            )
            
            bundle = HPVDInputBundle(
                trajectory=trajectory,
                dna=regime.dna_signature + self.rng.randn(self.K).astype(np.float32) * 0.01,  # Small DNA noise
                geometry_context={'LTV': 0.3, 'LVC': 0.1, 'K': 5.0},
                metadata={
                    'scenario': 'A',
                    'regime_id': regime_id,
                    'trajectory_id': f'scenario_A_hist_{i:03d}',
                    'trajectory_horizon': str(self.T),
                    'state_dim': str(self.D),
                    'dna_version': 'v1',
                    'schema_version': 'hpvd_input_v1',
                    'timestamp': (self.base_date + timedelta(days=i)).isoformat()
                }
            )
            historical.append(bundle)
        
        # Query trajectory (same regime)
        query_trajectory = self._generate_trajectory_from_regime(regime, noise_level=noise_level)
        query_bundle = HPVDInputBundle(
            trajectory=query_trajectory,
            dna=regime.dna_signature + self.rng.randn(self.K).astype(np.float32) * 0.01,
            geometry_context={'LTV': 0.3, 'LVC': 0.1, 'K': 5.0},
            metadata={
                'scenario': 'A',
                'regime_id': regime_id,
                'trajectory_id': 'scenario_A_query_001',
                'trajectory_horizon': str(self.T),
                'state_dim': str(self.D),
                'dna_version': 'v1',
                'schema_version': 'hpvd_input_v1',
                'timestamp': (self.base_date + timedelta(days=1000)).isoformat()
            }
        )
        
        return {
            'historical': historical,
            'query': [query_bundle]
        }
    
    def generate_scenario_b(self,
                           n_historical: int = 10) -> Dict[str, List[HPVDInputBundle]]:
        """
        Scenario B: Surface Similarity, Different Evolution
        
        Two trajectories look similar at the end, but:
        - One is entering compression (R3)
        - One is exiting compression (R1)
        - Cognitive DNA must differ
        
        Expected HPVD behavior:
        - Must NOT group them
        - Must separate into different families
        - Or reject one as non-analog
        """
        # Historical: R1 (stable expansion) - looks similar at end
        r1_trajectory = self._generate_trajectory_from_regime(self.regimes['R1'])
        # Make end state similar to R3
        r1_trajectory[-5:, :] = r1_trajectory[-5:, :] * 0.5  # Scale down end
        
        r1_bundle = HPVDInputBundle(
            trajectory=r1_trajectory,
            dna=self.regimes['R1'].dna_signature,
            geometry_context={'LTV': 0.3, 'LVC': 0.1, 'K': 5.0},
            metadata={
                'scenario': 'B',
                'regime_id': 'R1',
                'trajectory_id': 'scenario_B_hist_R1',
                'trajectory_horizon': str(self.T),
                'state_dim': str(self.D),
                'dna_version': 'v1',
                'schema_version': 'hpvd_input_v1',
                'timestamp': (self.base_date + timedelta(days=2000)).isoformat()
            }
        )
        
        # Query: R3 (compression) - similar end state but different evolution
        r3_trajectory = self._generate_trajectory_from_regime(self.regimes['R3'])
        # Make end state similar to R1
        r3_trajectory[-5:, :] = r3_trajectory[-5:, :] * 0.5  # Scale down end
        
        query_bundle = HPVDInputBundle(
            trajectory=r3_trajectory,
            dna=self.regimes['R3'].dna_signature,  # Different DNA!
            geometry_context={'LTV': 0.6, 'LVC': 0.3, 'K': 7.5},
            metadata={
                'scenario': 'B',
                'regime_id': 'R3',
                'trajectory_id': 'scenario_B_query_R3',
                'trajectory_horizon': str(self.T),
                'state_dim': str(self.D),
                'dna_version': 'v1',
                'schema_version': 'hpvd_input_v1',
                'timestamp': (self.base_date + timedelta(days=2100)).isoformat()
            }
        )
        
        # Add more R3 historical for context
        r3_historical = []
        for i in range(n_historical - 1):
            traj = self._generate_trajectory_from_regime(self.regimes['R3'])
            bundle = HPVDInputBundle(
                trajectory=traj,
                dna=self.regimes['R3'].dna_signature,
                geometry_context={'LTV': 0.6, 'LVC': 0.3, 'K': 7.5},
                metadata={
                    'scenario': 'B',
                    'regime_id': 'R3',
                    'trajectory_id': f'scenario_B_hist_R3_{i:03d}',
                    'trajectory_horizon': str(self.T),
                    'state_dim': str(self.D),
                    'dna_version': 'v1',
                    'schema_version': 'hpvd_input_v1',
                    'timestamp': (self.base_date + timedelta(days=2001 + i)).isoformat()
                }
            )
            r3_historical.append(bundle)
        
        return {
            'historical': [r1_bundle] + r3_historical,
            'query': [query_bundle]
        }
    
    def generate_scenario_c(self,
                           n_historical: int = 15,
                           scale_factors: List[float] = None) -> Dict[str, List[HPVDInputBundle]]:
        """
        Scenario C: Same Phase, Different Amplitude
        
        Same evolutionary structure, different scales.
        DNA is same phase.
        
        Expected behavior:
        - Grouped as analogs
        - Possibly lower confidence, but admissible
        """
        if scale_factors is None:
            scale_factors = [0.5, 1.0, 1.5, 2.0]
        
        regime = self.regimes['R1']
        historical = []
        
        for i, scale in enumerate(scale_factors):
            for j in range(n_historical // len(scale_factors)):
                trajectory = self._generate_trajectory_from_regime(
                    regime, scale=scale, noise_level=0.1
                )
                bundle = HPVDInputBundle(
                    trajectory=trajectory,
                    dna=regime.dna_signature,  # Same DNA
                    geometry_context={'LTV': 0.3 * scale, 'LVC': 0.1 * scale, 'K': 5.0},
                    metadata={
                        'scenario': 'C',
                        'regime_id': 'R1',
                        'scale': str(scale),
                        'trajectory_id': f'scenario_C_hist_scale{scale}_{j:03d}',
                        'trajectory_horizon': str(self.T),
                        'state_dim': str(self.D),
                        'dna_version': 'v1',
                        'schema_version': 'hpvd_input_v1',
                        'timestamp': (self.base_date + timedelta(days=3000 + i * 10 + j)).isoformat()
                    }
                )
                historical.append(bundle)
        
        # Query: medium scale
        query_trajectory = self._generate_trajectory_from_regime(regime, scale=1.0)
        query_bundle = HPVDInputBundle(
            trajectory=query_trajectory,
            dna=regime.dna_signature,
            geometry_context={'LTV': 0.3, 'LVC': 0.1, 'K': 5.0},
            metadata={
                'scenario': 'C',
                'regime_id': 'R1',
                'trajectory_id': 'scenario_C_query',
                'trajectory_horizon': str(self.T),
                'state_dim': str(self.D),
                'dna_version': 'v1',
                'schema_version': 'hpvd_input_v1',
                'timestamp': (self.base_date + timedelta(days=3100)).isoformat()
            }
        )
        
        return {
            'historical': historical,
            'query': [query_bundle]
        }
    
    def generate_scenario_d(self,
                           n_stable: int = 10,
                           n_stress: int = 10) -> Dict[str, List[HPVDInputBundle]]:
        """
        Scenario D: Transitional Phase (Ambiguous Case)
        
        Current trajectory is between stable and stress regimes.
        Historical data contains both clean examples.
        
        Expected behavior:
        - HPVD retrieves BOTH
        - Forms MULTIPLE analog families
        - No forced collapse into one answer
        """
        # Historical: Clean R1 (stable)
        r1_historical = []
        for i in range(n_stable):
            traj = self._generate_trajectory_from_regime(self.regimes['R1'])
            bundle = HPVDInputBundle(
                trajectory=traj,
                dna=self.regimes['R1'].dna_signature,
                geometry_context={'LTV': 0.3, 'LVC': 0.1, 'K': 5.0},
                metadata={
                    'scenario': 'D',
                    'regime_id': 'R1',
                    'trajectory_id': f'scenario_D_hist_R1_{i:03d}',
                    'trajectory_horizon': str(self.T),
                    'state_dim': str(self.D),
                    'dna_version': 'v1',
                    'schema_version': 'hpvd_input_v1',
                    'timestamp': (self.base_date + timedelta(days=4000 + i)).isoformat()
                }
            )
            r1_historical.append(bundle)
        
        # Historical: Clean R5 (stress)
        r5_historical = []
        for i in range(n_stress):
            traj = self._generate_trajectory_from_regime(self.regimes['R5'])
            bundle = HPVDInputBundle(
                trajectory=traj,
                dna=self.regimes['R5'].dna_signature,
                geometry_context={'LTV': 0.8, 'LVC': 0.5, 'K': 9.0},
                metadata={
                    'scenario': 'D',
                    'regime_id': 'R5',
                    'trajectory_id': f'scenario_D_hist_R5_{i:03d}',
                    'trajectory_horizon': str(self.T),
                    'state_dim': str(self.D),
                    'dna_version': 'v1',
                    'schema_version': 'hpvd_input_v1',
                    'timestamp': (self.base_date + timedelta(days=4100 + i)).isoformat()
                }
            )
            r5_historical.append(bundle)
        
        # Query: Transitional (R4 - between R1 and R5)
        query_trajectory = self._generate_trajectory_from_regime(self.regimes['R4'])
        # Blend characteristics
        query_trajectory = 0.5 * query_trajectory + 0.25 * self._generate_trajectory_from_regime(self.regimes['R1']) + \
                          0.25 * self._generate_trajectory_from_regime(self.regimes['R5'])
        
        # Blend DNA
        blended_dna = 0.5 * self.regimes['R4'].dna_signature + \
                     0.25 * self.regimes['R1'].dna_signature + \
                     0.25 * self.regimes['R5'].dna_signature
        
        query_bundle = HPVDInputBundle(
            trajectory=query_trajectory,
            dna=blended_dna,
            geometry_context={'LTV': 0.55, 'LVC': 0.3, 'K': 7.0},  # Between R1 and R5
            metadata={
                'scenario': 'D',
                'regime_id': 'R4',
                'trajectory_id': 'scenario_D_query_transitional',
                'trajectory_horizon': str(self.T),
                'state_dim': str(self.D),
                'dna_version': 'v1',
                'schema_version': 'hpvd_input_v1',
                'timestamp': (self.base_date + timedelta(days=4200)).isoformat()
            }
        )
        
        return {
            'historical': r1_historical + r5_historical,
            'query': [query_bundle]
        }
    
    def generate_scenario_e(self,
                           n_historical: int = 20) -> Dict[str, List[HPVDInputBundle]]:
        """
        Scenario E: Novel Structure (No Analogs Exist)
        
        Current trajectory uses unseen geometry/phase.
        No historical match exists.
        
        Expected behavior:
        - HPVD returns empty or very weak analog set
        - Explicit "low support" signal
        """
        # Historical: Only R1, R2, R3 (no R6)
        historical = []
        for regime_idx, regime_id in enumerate(['R1', 'R2', 'R3']):
            regime = self.regimes[regime_id]
            for i in range(n_historical // 3):
                traj = self._generate_trajectory_from_regime(regime)
                bundle = HPVDInputBundle(
                    trajectory=traj,
                    dna=regime.dna_signature,
                    geometry_context={'LTV': 0.3, 'LVC': 0.1, 'K': 5.0},
                    metadata={
                        'scenario': 'E',
                        'regime_id': regime_id,
                        'trajectory_id': f'scenario_E_hist_{regime_id}_{i:03d}',
                        'trajectory_horizon': str(self.T),
                        'state_dim': str(self.D),
                        'dna_version': 'v1',
                        'schema_version': 'hpvd_input_v1',
                        'timestamp': (self.base_date + timedelta(days=5000 + regime_idx * 100 + i)).isoformat()
                    }
                )
                historical.append(bundle)
        
        # Query: R6 (novel, unseen)
        query_trajectory = self._generate_trajectory_from_regime(self.regimes['R6'])
        query_bundle = HPVDInputBundle(
            trajectory=query_trajectory,
            dna=self.regimes['R6'].dna_signature,  # Novel DNA
            geometry_context={'LTV': 1.0, 'LVC': 0.8, 'K': 10.0},  # Unseen geometry
            metadata={
                'scenario': 'E',
                'regime_id': 'R6',
                'trajectory_id': 'scenario_E_query_novel',
                'trajectory_horizon': str(self.T),
                'state_dim': str(self.D),
                'dna_version': 'v1',
                'schema_version': 'hpvd_input_v1',
                'timestamp': (self.base_date + timedelta(days=6000)).isoformat()
            }
        )
        
        return {
            'historical': historical,
            'query': [query_bundle]
        }
    
    def generate_all_scenarios(self) -> Dict[str, Dict[str, List[HPVDInputBundle]]]:
        """Generate all 5 scenarios"""
        return {
            'scenario_A': self.generate_scenario_a(),
            'scenario_B': self.generate_scenario_b(),
            'scenario_C': self.generate_scenario_c(),
            'scenario_D': self.generate_scenario_d(),
            'scenario_E': self.generate_scenario_e()
        }
