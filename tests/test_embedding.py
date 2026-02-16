"""
Tests for EmbeddingComputer (PCA-based trajectory embedding)
"""

import os
import pytest
import numpy as np

from src.hpvd.embedding import EmbeddingComputer


class TestEmbeddingComputer:
    """Unit tests for EmbeddingComputer"""

    @pytest.fixture
    def sample_matrices(self):
        """Generate synthetic trajectory matrices for testing."""
        rng = np.random.RandomState(42)
        return rng.randn(50, 60, 45).astype(np.float32)

    @pytest.fixture
    def fitted_computer(self, sample_matrices):
        """Return an EmbeddingComputer already fitted on sample data."""
        computer = EmbeddingComputer(n_components=256)
        computer.fit(sample_matrices)
        return computer

    def test_fit_and_transform(self, sample_matrices):
        """Fit PCA, transform single matrix, verify output shape."""
        computer = EmbeddingComputer(n_components=256)
        computer.fit(sample_matrices)

        single = sample_matrices[0]  # (60, 45)
        embedding = computer.transform(single)

        assert embedding.shape == (256,)
        assert embedding.dtype == np.float32
        assert computer.is_fitted

    def test_transform_batch(self, fitted_computer, sample_matrices):
        """Batch transform N matrices, verify output shape."""
        embeddings = fitted_computer.transform_batch(sample_matrices)

        assert embeddings.shape == (50, 256)
        assert embeddings.dtype == np.float32

    def test_explained_variance(self, fitted_computer):
        """Explained variance ratio should be positive and <= 1."""
        evr = fitted_computer.explained_variance_ratio
        assert evr > 0.0
        assert evr <= 1.0

    def test_transform_reduces_dimension(self, fitted_computer):
        """2700-dim input (60*45) should produce 256-dim output."""
        rng = np.random.RandomState(99)
        matrix = rng.randn(60, 45).astype(np.float32)

        embedding = fitted_computer.transform(matrix)
        assert embedding.shape[0] == 256
        assert matrix.flatten().shape[0] == 2700

    def test_save_and_load(self, fitted_computer, sample_matrices, tmp_path):
        """Save PCA, load it, verify identical transform results."""
        path = str(tmp_path / "pca_model.pkl")
        fitted_computer.save(path)

        loaded = EmbeddingComputer(n_components=256)
        loaded.load(path)

        assert loaded.is_fitted

        matrix = sample_matrices[0]
        emb_original = fitted_computer.transform(matrix)
        emb_loaded = loaded.transform(matrix)

        np.testing.assert_array_almost_equal(emb_original, emb_loaded, decimal=5)

    def test_not_fitted_raises(self):
        """Transform before fit should raise RuntimeError."""
        computer = EmbeddingComputer(n_components=256)

        assert not computer.is_fitted

        with pytest.raises(RuntimeError, match="not fitted"):
            computer.transform(np.random.randn(60, 45).astype(np.float32))

        with pytest.raises(RuntimeError, match="not fitted"):
            computer.transform_batch(np.random.randn(5, 60, 45).astype(np.float32))

        with pytest.raises(RuntimeError, match="not fitted"):
            computer.save("/tmp/should_not_exist.pkl")

    def test_deterministic(self, fitted_computer):
        """Same input should always produce exactly the same embedding."""
        rng = np.random.RandomState(123)
        matrix = rng.randn(60, 45).astype(np.float32)

        emb1 = fitted_computer.transform(matrix)
        emb2 = fitted_computer.transform(matrix)

        np.testing.assert_array_equal(emb1, emb2)

    def test_few_samples_capped_components(self):
        """When N < n_components, PCA caps and pads to requested dim."""
        rng = np.random.RandomState(7)
        small_data = rng.randn(10, 60, 45).astype(np.float32)

        computer = EmbeddingComputer(n_components=256)
        computer.fit(small_data)

        embedding = computer.transform(small_data[0])
        assert embedding.shape == (256,)
        assert computer.is_fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
