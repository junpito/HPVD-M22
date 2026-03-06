"""
Document Retrieval Strategy
============================

Semantic vector search for chatbot / banking / document domains.
Uses ``sentence-transformers`` for 384-d embeddings and FAISS inner-product
index (cosine similarity on L2-normalized vectors).

Architecture mirrors HPVD core:
    - ``topic`` ≡ regime   (categorical pre-filter via inverted index)
    - ``doc_type`` ≡ DNA   (categorical match boost)
    - ``cosine_similarity`` ≡ confidence  (calibrated 0–1 similarity)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import os
import pickle

import numpy as np

from ...family import (
    FamilyCoherence,
    StructuralSignature,
    UncertaintyFlags,
)
from ..retrieval_strategy import (
    FamilyAssignment,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalStrategy,
)


# ---------------------------------------------------------------------------
# Domain-specific data types
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """
    A document chunk — the document-domain equivalent of a ``Trajectory``.

    Attributes:
        chunk_id: Unique chunk identifier.
        text: Raw text content.
        topic: Categorical topic label (equivalent to regime).
        doc_type: Document type label (equivalent to DNA phase).
        metadata: Arbitrary metadata.
        embedding: Optional pre-computed embedding (384-d).
    """

    chunk_id: str
    text: str
    topic: str = ""
    doc_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class DocumentRetrievalConfig:
    """Configuration for ``DocumentRetrievalStrategy``."""

    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    default_k: int = 25
    min_similarity: float = 0.3
    doc_type_boost: float = 0.05  # additive boost when doc_type matches


class DocumentRetrievalStrategy(RetrievalStrategy):
    """
    Document-domain retrieval strategy using sentence-transformers + FAISS.

    Lazy-loads the sentence-transformer model on first use.

    Usage::

        strategy = DocumentRetrievalStrategy()
        strategy.build_index(chunks)  # List[DocumentChunk]
        result = strategy.search({"text": "refund policy", "allowed_topics": ["refund"]})
        families = strategy.compute_families(result.candidates)
    """

    def __init__(self, config: Optional[DocumentRetrievalConfig] = None):
        self._config = config or DocumentRetrievalConfig()
        self._model = None  # lazy
        self._chunks: List[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        # Topic inverted index: topic → set of chunk indices
        self._topic_index: Dict[str, Set[int]] = {}
        self._is_built = False

    # ------------------------------------------------------------------
    # RetrievalStrategy interface
    # ------------------------------------------------------------------

    @property
    def domain(self) -> str:
        return "document"

    def build_index(self, corpus: List[DocumentChunk]) -> None:  # type: ignore[override]
        """
        Build FAISS index and topic inverted index from ``DocumentChunk`` list.
        """
        import faiss  # local import to avoid hard dep at module level

        self._chunks = list(corpus)
        if not self._chunks:
            self._is_built = True
            return

        self._ensure_model_loaded()

        # Compute embeddings for chunks that don't already have one
        texts_to_encode: List[str] = []
        indices_to_encode: List[int] = []
        for i, chunk in enumerate(self._chunks):
            if chunk.embedding is None:
                texts_to_encode.append(chunk.text)
                indices_to_encode.append(i)

        if texts_to_encode:
            encoded = self._model.encode(
                texts_to_encode, show_progress_bar=False, convert_to_numpy=True
            )
            for j, idx in enumerate(indices_to_encode):
                self._chunks[idx].embedding = encoded[j].astype(np.float32)

        # Stack all embeddings and L2-normalize for cosine via inner product
        self._embeddings = np.vstack(
            [c.embedding for c in self._chunks]
        ).astype(np.float32)
        faiss.normalize_L2(self._embeddings)

        # Build FAISS IndexFlatIP (inner product on normalized = cosine)
        dim = self._embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(self._embeddings)

        # Build topic inverted index
        self._topic_index = {}
        for i, chunk in enumerate(self._chunks):
            topic = chunk.topic or "__none__"
            self._topic_index.setdefault(topic, set()).add(i)

        self._is_built = True

    def search(self, query: Dict[str, Any], k: int = 25) -> RetrievalResult:
        """
        Search for relevant document chunks.

        Expected *query* keys:
            - ``text`` (str): query text
            - ``allowed_topics`` (List[str], optional): topic pre-filter
            - ``allowed_doc_types`` (List[str], optional): doc-type filter
            - ``query_id`` (str, optional): identifier
        """
        import faiss  # noqa: F811

        if not self._is_built or not self._chunks:
            return RetrievalResult(
                candidates=[],
                diagnostics={"status": "empty_index"},
                query_id=query.get("query_id", ""),
            )

        self._ensure_model_loaded()

        text = query.get("text", "")
        allowed_topics: List[str] = query.get("allowed_topics", [])
        allowed_doc_types: List[str] = query.get("allowed_doc_types", [])
        query_id: str = query.get("query_id", "")

        # Embed & normalize query
        q_emb = self._model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        q_emb = q_emb.astype(np.float32)
        faiss.normalize_L2(q_emb)

        # Topic pre-filter
        if allowed_topics:
            allowed_indices: Set[int] = set()
            for t in allowed_topics:
                allowed_indices |= self._topic_index.get(t, set())
        else:
            allowed_indices = set(range(len(self._chunks)))

        if not allowed_indices:
            return RetrievalResult(
                candidates=[],
                diagnostics={"status": "no_topic_match", "allowed_topics": allowed_topics},
                query_id=query_id,
            )

        # Build IDSelector for filtered search
        id_array = np.array(sorted(allowed_indices), dtype=np.int64)
        sel = faiss.IDSelectorArray(id_array)
        params = faiss.SearchParametersIVF()
        params.sel = sel
        search_k = min(k, len(allowed_indices))

        # FAISS inner-product search
        try:
            scores, ids = self._faiss_index.search(
                q_emb, search_k, params=params
            )
        except TypeError:
            # Fallback for FAISS versions without params in IndexFlatIP.search
            scores, ids = self._faiss_index.search(q_emb, len(self._chunks))
            # Manual filter
            mask = np.isin(ids[0], id_array)
            scores = scores[0][mask][:search_k]
            ids_filtered = ids[0][mask][:search_k]
            scores = scores.reshape(1, -1)
            ids = ids_filtered.reshape(1, -1)

        candidates: List[RetrievalCandidate] = []
        for j in range(ids.shape[1]):
            idx = int(ids[0, j])
            if idx < 0:
                continue
            sim = float(scores[0, j])
            chunk = self._chunks[idx]

            # Doc-type boost
            if allowed_doc_types and chunk.doc_type in allowed_doc_types:
                sim += self._config.doc_type_boost

            # Clip to [0, 1]
            sim = max(0.0, min(sim, 1.0))

            if sim < self._config.min_similarity:
                continue

            candidates.append(
                RetrievalCandidate(
                    candidate_id=chunk.chunk_id,
                    score=sim,
                    metadata={
                        "topic": chunk.topic,
                        "doc_type": chunk.doc_type,
                        "text_preview": chunk.text[:120],
                    },
                    source_domain="document",
                )
            )

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:k]

        return RetrievalResult(
            candidates=candidates,
            diagnostics={
                "total_chunks": len(self._chunks),
                "topic_filtered": len(allowed_indices),
                "returned": len(candidates),
            },
            query_id=query_id,
        )

    def compute_families(
        self, candidates: List[RetrievalCandidate]
    ) -> List[FamilyAssignment]:
        """
        Group candidates into families by ``topic``.

        Each unique topic becomes a family.  Coherence and uncertainty
        are computed identically to HPVD core (mean score, dispersion,
        weak_support flag for small groups).
        """
        if not candidates:
            return []

        # Group by topic
        topic_groups: Dict[str, List[RetrievalCandidate]] = {}
        for c in candidates:
            topic = c.metadata.get("topic", "__none__")
            topic_groups.setdefault(topic, []).append(c)

        assignments: List[FamilyAssignment] = []
        for idx, (topic, members) in enumerate(sorted(topic_groups.items()), start=1):
            scores = [m.score for m in members]
            mean_score = float(np.mean(scores))
            dispersion = float(np.std(scores)) if len(scores) > 1 else 0.0

            coherence = FamilyCoherence(
                mean_confidence=mean_score,
                dispersion=dispersion,
                size=len(members),
            )
            structural_signature = StructuralSignature(phase=topic)
            uncertainty_flags = UncertaintyFlags(
                phase_boundary=False,
                weak_support=(len(members) < 5 or dispersion > 0.3 or mean_score < 0.4),
                partial_overlap=False,
            )

            assignments.append(
                FamilyAssignment(
                    family_id=f"DF_{idx:03d}",
                    members=members,
                    coherence=coherence,
                    structural_signature=structural_signature,
                    uncertainty_flags=uncertainty_flags,
                )
            )

        return assignments

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save embeddings, chunk metadata, and FAISS index to *path*."""
        import faiss  # noqa: F811

        os.makedirs(path, exist_ok=True)
        # Save FAISS index
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, os.path.join(path, "faiss_index.bin"))
        # Save chunk metadata (without embeddings — those live in FAISS)
        meta = []
        for c in self._chunks:
            meta.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "topic": c.topic,
                "doc_type": c.doc_type,
                "metadata": c.metadata,
            })
        with open(os.path.join(path, "chunks_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        # Save embeddings matrix
        if self._embeddings is not None:
            np.save(os.path.join(path, "embeddings.npy"), self._embeddings)
        # Save config
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(self._config, f)

    def load(self, path: str) -> None:
        """Load from disk."""
        import faiss  # noqa: F811

        with open(os.path.join(path, "config.pkl"), "rb") as f:
            self._config = pickle.load(f)
        with open(os.path.join(path, "chunks_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self._embeddings = np.load(os.path.join(path, "embeddings.npy"))
        self._faiss_index = faiss.read_index(os.path.join(path, "faiss_index.bin"))

        # Reconstruct chunks
        self._chunks = []
        for i, m in enumerate(meta):
            self._chunks.append(
                DocumentChunk(
                    chunk_id=m["chunk_id"],
                    text=m["text"],
                    topic=m["topic"],
                    doc_type=m["doc_type"],
                    metadata=m["metadata"],
                    embedding=self._embeddings[i],
                )
            )
        # Rebuild topic index
        self._topic_index = {}
        for i, chunk in enumerate(self._chunks):
            topic = chunk.topic or "__none__"
            self._topic_index.setdefault(topic, set()).add(i)

        self._is_built = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._config.model_name)
