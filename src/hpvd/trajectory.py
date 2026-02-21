"""
Trajectory Data Model
=====================

Core trajectory entity representing 60 days × 45 features of financial data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import uuid


@dataclass
class HPVDInputBundle:
    """
    Outcome-blind HPVD input bundle (Matrix22 spec).

    This represents the **only** object that HPVD core is allowed to see.
    It contains:
        - trajectory: T × D state matrix (e.g. 60 × 45)
        - dna: compressed evolutionary phase vector (no outcomes)
        - geometry_context: descriptive structural metrics (no thresholds/decisions)
        - metadata: deterministic, replayable identifiers (no future info)

    Contract (frozen for MVP):
        - trajectory shape must be (T, D) with T > 0, D > 0
        - dna must be 1-D float array
        - geometry_context must be dict[str, float]
        - metadata must be dict[str, str]
    """

    trajectory: np.ndarray
    dna: np.ndarray
    geometry_context: Dict[str, float]
    metadata: Dict[str, str]

    # Forbidden outcome fields — if any key below appears in metadata,
    # validate() will reject the bundle.
    _OUTCOME_KEYS = frozenset({
        'label_h1', 'label_h5', 'return_h1', 'return_h5',
        'p_up', 'p_down', 'entropy', 'confidence_interval',
    })

    def validate(self) -> bool:
        """
        Validate bundle against the frozen MVP contract.

        Returns:
            True if valid.

        Raises:
            ValueError: with a descriptive message on the first violation found.
        """
        # --- trajectory ---
        if not isinstance(self.trajectory, np.ndarray):
            raise ValueError(f"trajectory must be np.ndarray, got {type(self.trajectory).__name__}")
        if self.trajectory.ndim != 2:
            raise ValueError(f"trajectory must be 2-D (T, D), got shape {self.trajectory.shape}")
        if self.trajectory.shape[0] == 0 or self.trajectory.shape[1] == 0:
            raise ValueError(f"trajectory dimensions must be > 0, got shape {self.trajectory.shape}")
        if np.isnan(self.trajectory).any():
            raise ValueError("trajectory contains NaN values")

        # --- dna ---
        if not isinstance(self.dna, np.ndarray):
            raise ValueError(f"dna must be np.ndarray, got {type(self.dna).__name__}")
        if self.dna.ndim != 1:
            raise ValueError(f"dna must be 1-D, got shape {self.dna.shape}")

        # --- geometry_context ---
        if not isinstance(self.geometry_context, dict):
            raise ValueError(f"geometry_context must be dict, got {type(self.geometry_context).__name__}")

        # --- metadata ---
        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be dict, got {type(self.metadata).__name__}")

        # --- outcome-blind guard ---
        leaked = self._OUTCOME_KEYS & set(self.metadata.keys())
        if leaked:
            raise ValueError(
                f"metadata contains outcome fields (violates outcome-blind contract): {leaked}"
            )

        return True


@dataclass
class Trajectory:
    """
    Core trajectory entity - 60 days × 45 features.

    NOTE (Matrix22):
        - This class currently still stores outcome fields (labels/returns)
          because of historical code and tests.
        - For HPVD-M22, outcomes belong to PMR/adapter layers, **not** HPVD core.
        - New HPVD logic should use `HPVDInputBundle` via `to_hpvd_input()`
          and must remain outcome-blind.

    Attributes:
        trajectory_id: Unique identifier (UUID)
        asset_id: Asset ticker (e.g., "AAPL", "BTC-USD")
        end_timestamp: End date of the 60-day window
        matrix: Raw feature matrix (60, 45)
        embedding: Reduced embedding for FAISS (256,)
        dna: Cognitive DNA vector encoding evolutionary phase (K-dim)
        label_h1: H1 outcome (+1 or -1)  # DEPRECATED for HPVD core
        label_h5: H5 outcome (+1 or -1)  # DEPRECATED for HPVD core
        return_h1: Actual H1 return (float)  # DEPRECATED for HPVD core
        return_h5: Actual H5 return (float)  # DEPRECATED for HPVD core
        trend_regime: Trend classification (-1, 0, +1)
        volatility_regime: Volatility classification (-1, 0, +1)
        structural_regime: Structure classification (-1, 0, +1)
        asset_class: Asset category
    """
    # Identity
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str = ""
    end_timestamp: datetime = field(default_factory=datetime.now)

    # Data
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((60, 45), dtype=np.float32))
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))
    dna: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))

    # Labels / outcomes (kept for backward-compat; outcome-blind HPVD must ignore)
    label_h1: int = 0
    label_h5: int = 0
    return_h1: float = 0.0
    return_h5: float = 0.0

    # Regimes
    trend_regime: int = 0
    volatility_regime: int = 0
    structural_regime: int = 0

    # Metadata
    asset_class: str = "equity"

    def get_regime_tuple(self) -> Tuple[int, int, int]:
        """Get regime as tuple for indexing"""
        return (self.trend_regime, self.volatility_regime, self.structural_regime)

    def get_flattened_matrix(self) -> np.ndarray:
        """Get flattened matrix (2700,)"""
        return self.matrix.flatten().astype(np.float32)

    def to_hpvd_input(
        self,
        dna: Optional[np.ndarray] = None,
        geometry_context: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> HPVDInputBundle:
        """
        Build an outcome-blind HPVD input bundle from this trajectory.

        Notes:
            - `dna` must encode evolutionary phase only (no outcomes).
            - `geometry_context` is descriptive only (no thresholds/decisions).
            - `metadata` must be deterministic and must not depend on future info.
        """
        # Use stored DNA if none provided; fall back to zero placeholder
        if dna is None:
            if np.any(self.dna != 0):
                dna = self.dna
            else:
                dna = np.zeros(16, dtype=np.float32)

        if geometry_context is None:
            geometry_context = {}

        if metadata is None:
            metadata = {
                "trajectory_horizon": str(self.matrix.shape[0]),
                "state_dim": str(self.matrix.shape[1]),
                "schema_version": "hpvd_input_v1",
                "trajectory_id": self.trajectory_id,
                "asset_id": self.asset_id,
                "timestamp": self.end_timestamp.isoformat(),
            }

        return HPVDInputBundle(
            trajectory=self.matrix.astype(np.float32),
            dna=dna.astype(np.float32),
            geometry_context=geometry_context,
            metadata=metadata,
        )

    def validate(self) -> bool:
        """
        Validate trajectory data integrity.

        Matrix22 note:
            - Validation focuses on geometric/structural invariants.
            - Outcome fields (labels/returns) are NOT part of HPVD validity
              and will gradually be removed from this check.
        """
        if self.matrix.shape != (60, 45):
            return False
        if self.embedding.shape != (256,):
            return False
        # Outcome fields are intentionally NOT enforced for HPVD core validity.
        if self.trend_regime not in [-1, 0, 1]:
            return False
        if self.volatility_regime not in [-1, 0, 1]:
            return False
        if self.structural_regime not in [-1, 0, 1]:
            return False
        if np.isnan(self.matrix).any():
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"Trajectory(id={self.trajectory_id[:8]}..., "
            f"asset={self.asset_id}, "
            f"date={self.end_timestamp.strftime('%Y-%m-%d')}, "
            f"regime={self.get_regime_tuple()})"
        )

