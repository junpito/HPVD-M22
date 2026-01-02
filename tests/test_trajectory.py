"""
Tests for Trajectory class
"""

import pytest
import numpy as np
from datetime import datetime

from src.hpvd.trajectory import Trajectory


class TestTrajectory:
    """Unit tests for Trajectory class"""
    
    def test_create_default_trajectory(self):
        """Test creating trajectory with default values"""
        traj = Trajectory()
        
        assert traj.trajectory_id is not None
        assert traj.asset_id == ""
        assert traj.matrix.shape == (60, 45)
        assert traj.embedding.shape == (256,)
    
    def test_validate_valid_trajectory(self):
        """Test validation with valid trajectory"""
        traj = Trajectory(
            trajectory_id="test_1",
            asset_id="AAPL",
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
            label_h1=1,
            label_h5=-1,
            trend_regime=1,
            volatility_regime=0,
            structural_regime=-1
        )
        
        assert traj.validate() == True
    
    def test_validate_invalid_matrix_shape(self):
        """Test validation with wrong matrix shape"""
        traj = Trajectory(
            trajectory_id="test_1",
            asset_id="AAPL",
            matrix=np.random.randn(50, 45).astype(np.float32),  # Wrong shape
            embedding=np.random.randn(256).astype(np.float32),
            label_h1=1,
            label_h5=-1,
            trend_regime=1,
            volatility_regime=0,
            structural_regime=-1
        )
        
        assert traj.validate() == False
    
    def test_validate_invalid_embedding_shape(self):
        """Test validation with wrong embedding shape"""
        traj = Trajectory(
            trajectory_id="test_1",
            asset_id="AAPL",
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(128).astype(np.float32),  # Wrong shape
            label_h1=1,
            label_h5=-1,
            trend_regime=1,
            volatility_regime=0,
            structural_regime=-1
        )
        
        assert traj.validate() == False
    
    def test_validate_invalid_regime(self):
        """Test validation with invalid regime value"""
        traj = Trajectory(
            trajectory_id="test_1",
            asset_id="AAPL",
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
            label_h1=1,
            label_h5=-1,
            trend_regime=5,  # Invalid regime
            volatility_regime=0,
            structural_regime=-1
        )
        
        assert traj.validate() == False
    
    def test_validate_nan_matrix(self):
        """Test validation with NaN in matrix"""
        matrix = np.random.randn(60, 45).astype(np.float32)
        matrix[0, 0] = np.nan
        
        traj = Trajectory(
            trajectory_id="test_1",
            asset_id="AAPL",
            matrix=matrix,
            embedding=np.random.randn(256).astype(np.float32),
            label_h1=1,
            label_h5=-1,
            trend_regime=1,
            volatility_regime=0,
            structural_regime=-1
        )
        
        assert traj.validate() == False
    
    def test_get_regime_tuple(self):
        """Test regime tuple getter"""
        traj = Trajectory(
            trend_regime=1,
            volatility_regime=-1,
            structural_regime=0
        )
        
        assert traj.get_regime_tuple() == (1, -1, 0)
    
    def test_get_flattened_matrix(self):
        """Test matrix flattening"""
        traj = Trajectory(
            matrix=np.ones((60, 45), dtype=np.float32)
        )
        
        flat = traj.get_flattened_matrix()
        
        assert flat.shape == (2700,)
        assert flat.dtype == np.float32
        assert np.all(flat == 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

