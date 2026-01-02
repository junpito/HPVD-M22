"""
Tests for SparseRegimeIndex
"""

import pytest
from src.hpvd.sparse_index import SparseRegimeIndex


class TestSparseRegimeIndex:
    """Unit tests for SparseRegimeIndex"""
    
    @pytest.fixture
    def index(self):
        """Create test index with sample data"""
        idx = SparseRegimeIndex()
        idx.add("t1", trend=1, volatility=0, structural=1, asset_id="AAPL", asset_class="equity")
        idx.add("t2", trend=1, volatility=1, structural=1, asset_id="AAPL", asset_class="equity")
        idx.add("t3", trend=-1, volatility=0, structural=0, asset_id="MSFT", asset_class="equity")
        idx.add("t4", trend=0, volatility=-1, structural=1, asset_id="BTC", asset_class="crypto")
        return idx
    
    def test_add_trajectory(self, index):
        """Test adding trajectories"""
        assert index.total_count == 4
        assert len(index.trajectory_regimes) == 4
    
    def test_filter_exact_match(self, index):
        """Test exact regime filtering"""
        result = index.filter_by_regime(trend=1, volatility=0, structural=1, allow_adjacent=False)
        assert result == {"t1"}
    
    def test_filter_with_adjacent(self, index):
        """Test regime filtering with adjacent regimes"""
        result = index.filter_by_regime(trend=1, volatility=0, structural=1, allow_adjacent=True)
        assert "t1" in result
        assert "t2" in result  # Adjacent volatility (0 vs 1)
    
    def test_filter_by_asset(self, index):
        """Test filtering by asset"""
        result = index.filter_by_asset(["AAPL"])
        assert result == {"t1", "t2"}
    
    def test_filter_by_asset_class(self, index):
        """Test filtering by asset class"""
        result = index.filter_by_asset_class(["crypto"])
        assert result == {"t4"}
    
    def test_combined_filter(self, index):
        """Test combined filtering"""
        result = index.combined_filter(
            trend=1,
            asset_classes=["equity"],
            allow_adjacent=True
        )
        assert "t1" in result
        assert "t2" in result
        assert "t4" not in result  # crypto
    
    def test_regime_match_score_exact(self, index):
        """Test exact regime match score"""
        score = index.get_regime_match_score((1, 0, 1), "t1")
        assert score == 1.0
    
    def test_regime_match_score_adjacent(self, index):
        """Test adjacent regime match score"""
        score = index.get_regime_match_score((1, 0, 1), "t2")
        # t2 has (1, 1, 1), query is (1, 0, 1)
        # trend: exact (1.0), vol: adjacent (0.5), struct: exact (1.0)
        expected = (1.0 + 0.5 + 1.0) / 3
        assert abs(score - expected) < 0.01
    
    def test_regime_match_score_not_found(self, index):
        """Test regime match score for non-existent trajectory"""
        score = index.get_regime_match_score((1, 0, 1), "nonexistent")
        assert score == 0.0
    
    def test_remove_trajectory(self, index):
        """Test removing trajectory"""
        index.remove("t1")
        
        assert index.total_count == 3
        assert "t1" not in index.trajectory_regimes
        
        result = index.filter_by_regime(trend=1, volatility=0, structural=1, allow_adjacent=False)
        assert result == set()
    
    def test_get_statistics(self, index):
        """Test statistics generation"""
        stats = index.get_statistics()
        
        assert stats['total_trajectories'] == 4
        assert stats['unique_assets'] == 3
        assert stats['unique_asset_classes'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

