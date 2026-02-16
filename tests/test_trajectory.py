"""
Tests for Trajectory class
"""

import pytest
import numpy as np
from datetime import datetime

from src.hpvd.trajectory import Trajectory, HPVDInputBundle


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

    def test_to_hpvd_input_default(self):
        """HPVDInputBundle is outcome-blind and structurally valid with defaults."""
        traj = Trajectory(
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
            trend_regime=1,
            volatility_regime=0,
            structural_regime=-1,
        )

        bundle = traj.to_hpvd_input()

        assert isinstance(bundle, HPVDInputBundle)
        # Trajectory geometry
        assert bundle.trajectory.shape == (60, 45)
        assert bundle.trajectory.dtype == np.float32
        # DNA placeholder
        assert bundle.dna.shape[0] == 16
        # Metadata invariants
        assert bundle.metadata["trajectory_horizon"] == "60"
        assert bundle.metadata["state_dim"] == "45"
        assert bundle.metadata["schema_version"] == "hpvd_input_v1"
        # Ensure no outcome fields leak into bundle
        serialized = {**bundle.geometry_context, **bundle.metadata}
        for forbidden in ["label_h1", "label_h5", "return_h1", "return_h5"]:
            assert forbidden not in serialized

    def test_to_hpvd_input_custom_dna_and_context(self):
        """Custom DNA and geometry_context are propagated exactly."""
        traj = Trajectory(
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
        )

        dna = np.ones(8, dtype=np.float32)
        geometry_context = {"LTV": 0.42, "LVC": 0.18}
        metadata = {
            "trajectory_horizon": "60",
            "state_dim": "45",
            "schema_version": "hpvd_input_v1",
            "custom": "yes",
        }

        bundle = traj.to_hpvd_input(dna=dna, geometry_context=geometry_context, metadata=metadata)

        assert bundle.dna.shape == (8,)
        assert np.allclose(bundle.dna, dna)
        assert bundle.geometry_context == geometry_context
        assert bundle.metadata["custom"] == "yes"


    def test_trajectory_with_dna(self):
        """DNA field is stored and accessible on Trajectory."""
        dna = np.array([1.0, 0.5, -0.3, 0.0] * 4, dtype=np.float32)
        traj = Trajectory(
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
            dna=dna,
        )

        assert traj.dna.shape == (16,)
        np.testing.assert_array_equal(traj.dna, dna)

    def test_trajectory_default_dna(self):
        """Default DNA is a zero vector."""
        traj = Trajectory()
        assert traj.dna.shape == (16,)
        assert np.all(traj.dna == 0)

    def test_to_hpvd_input_preserves_stored_dna(self):
        """to_hpvd_input() uses self.dna when no explicit DNA is passed."""
        stored_dna = np.ones(16, dtype=np.float32) * 0.42
        traj = Trajectory(
            matrix=np.random.randn(60, 45).astype(np.float32),
            embedding=np.random.randn(256).astype(np.float32),
            dna=stored_dna,
        )

        bundle = traj.to_hpvd_input()

        np.testing.assert_array_almost_equal(bundle.dna, stored_dna)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

