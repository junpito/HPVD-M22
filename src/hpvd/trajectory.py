"""
Trajectory Data Model
=====================

Core trajectory entity representing 60 days × 45 features of financial data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import uuid


@dataclass
class Trajectory:
    """
    Core trajectory entity - 60 days × 45 features
    
    Attributes:
        trajectory_id: Unique identifier (UUID)
        asset_id: Asset ticker (e.g., "AAPL", "BTC-USD")
        end_timestamp: End date of the 60-day window
        matrix: Raw feature matrix (60, 45)
        embedding: Reduced embedding for FAISS (256,)
        label_h1: H1 outcome (+1 or -1)
        label_h5: H5 outcome (+1 or -1)
        return_h1: Actual H1 return (float)
        return_h5: Actual H5 return (float)
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
    
    # Labels
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
    
    def validate(self) -> bool:
        """Validate trajectory data integrity"""
        if self.matrix.shape != (60, 45):
            return False
        if self.embedding.shape != (256,):
            return False
        if self.label_h1 not in [-1, 0, 1]:
            return False
        if self.label_h5 not in [-1, 0, 1]:
            return False
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

