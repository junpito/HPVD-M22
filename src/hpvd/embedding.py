"""
Embedding Computer
==================

PCA-based dimensionality reduction for trajectory matrices.
Transforms 60x45 (2700-dim) trajectory matrices into compact
256-dim embeddings suitable for FAISS nearest-neighbor search.
"""

from typing import Optional
import numpy as np
import pickle

from sklearn.decomposition import PCA


class EmbeddingComputer:
    """
    Compute dense embeddings from trajectory matrices using PCA.

    Reduces 60x45 trajectory matrices (2700 features) to compact
    n_components-dimensional vectors that preserve the principal
    variance structure of the data.

    Usage:
        computer = EmbeddingComputer(n_components=256)
        computer.fit(matrices)                      # (N, 60, 45)
        embedding = computer.transform(matrix)      # (60, 45) -> (256,)
        embeddings = computer.transform_batch(mats) # (N, 60, 45) -> (N, 256)
    """

    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self._pca: Optional[PCA] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether PCA has been fitted on training data."""
        return self._is_fitted

    @property
    def explained_variance_ratio(self) -> float:
        """Cumulative explained variance ratio (0-1) of the fitted PCA."""
        if not self._is_fitted:
            return 0.0
        return float(self._pca.explained_variance_ratio_.sum())

    def fit(self, matrices: np.ndarray) -> None:
        """
        Fit PCA on a collection of trajectory matrices.

        Args:
            matrices: (N, T, D) array of trajectory matrices,
                      e.g. (N, 60, 45).
        """
        n = matrices.shape[0]
        flattened = matrices.reshape(n, -1).astype(np.float64)

        # Cap n_components to the number of samples or features
        max_components = min(n, flattened.shape[1])
        actual_components = min(self.n_components, max_components)

        self._pca = PCA(n_components=actual_components)
        self._pca.fit(flattened)
        self._is_fitted = True

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        Transform a single trajectory matrix into an embedding.

        Args:
            matrix: (T, D) trajectory matrix, e.g. (60, 45).

        Returns:
            (n_components,) embedding vector as float32.
        """
        if not self._is_fitted:
            raise RuntimeError("EmbeddingComputer not fitted. Call fit() first.")

        flattened = matrix.flatten().reshape(1, -1).astype(np.float64)
        embedding = self._pca.transform(flattened)[0]

        # Pad if PCA produced fewer components than requested
        if len(embedding) < self.n_components:
            embedding = np.pad(
                embedding,
                (0, self.n_components - len(embedding)),
                mode="constant",
            )

        return embedding.astype(np.float32)

    def transform_batch(self, matrices: np.ndarray) -> np.ndarray:
        """
        Transform a batch of trajectory matrices into embeddings.

        Args:
            matrices: (N, T, D) array of trajectory matrices.

        Returns:
            (N, n_components) array of embeddings as float32.
        """
        if not self._is_fitted:
            raise RuntimeError("EmbeddingComputer not fitted. Call fit() first.")

        n = matrices.shape[0]
        flattened = matrices.reshape(n, -1).astype(np.float64)
        embeddings = self._pca.transform(flattened)

        # Pad if PCA produced fewer components than requested
        actual_dim = embeddings.shape[1]
        if actual_dim < self.n_components:
            pad_width = ((0, 0), (0, self.n_components - actual_dim))
            embeddings = np.pad(embeddings, pad_width, mode="constant")

        return embeddings.astype(np.float32)

    def save(self, path: str) -> None:
        """Save fitted PCA model to disk."""
        if not self._is_fitted:
            raise RuntimeError("EmbeddingComputer not fitted. Nothing to save.")

        data = {
            "pca": self._pca,
            "n_components": self.n_components,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load a previously saved PCA model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._pca = data["pca"]
        self.n_components = data["n_components"]
        self._is_fitted = True
