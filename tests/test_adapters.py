"""
Tests for the HPVD Adapter Layer
=================================

Covers:
    - RetrievalStrategy ABC enforcement
    - DocumentRetrievalStrategy (sentence-transformers + FAISS)

Remaining tests after Manithy v1 refactoring (removed Finance, J-file,
Pipeline, and Dispatcher tests).
"""

import json

import numpy as np
import pytest

from src.hpvd.adapters.family_types import FamilyCoherence, StructuralSignature, UncertaintyFlags

from src.hpvd.adapters.retrieval_strategy import (
    FamilyAssignment,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalStrategy,
)
from src.hpvd.adapters.strategies.document_strategy import (
    DocumentChunk,
    DocumentRetrievalConfig,
    DocumentRetrievalStrategy,
)


# =====================================================================
# Helpers / fixtures
# =====================================================================


@pytest.fixture
def document_chunks():
    """Small corpus of document chunks for document-strategy tests."""
    return [
        DocumentChunk(chunk_id="c1", text="How do I request a refund for my order?", topic="refund", doc_type="faq"),
        DocumentChunk(chunk_id="c2", text="Refund processing takes 3–5 business days.", topic="refund", doc_type="policy"),
        DocumentChunk(chunk_id="c3", text="What is the interest rate on personal loans?", topic="loan", doc_type="faq"),
        DocumentChunk(chunk_id="c4", text="Loan applications require a credit check.", topic="loan", doc_type="policy"),
        DocumentChunk(chunk_id="c5", text="How to open a savings account online.", topic="banking", doc_type="guide"),
        DocumentChunk(chunk_id="c6", text="Mobile banking app features and setup.", topic="banking", doc_type="guide"),
        DocumentChunk(chunk_id="c7", text="Return merchandise authorization process explained.", topic="refund", doc_type="guide"),
        DocumentChunk(chunk_id="c8", text="Overdraft protection and fee schedule.", topic="banking", doc_type="policy"),
    ]


@pytest.fixture
def doc_strategy():
    return DocumentRetrievalStrategy(DocumentRetrievalConfig(min_similarity=0.0))


# =====================================================================
# TestRetrievalStrategyInterface  (~3 tests)
# =====================================================================


class TestRetrievalStrategyInterface:
    """Verify the ABC contract."""

    def test_abc_cannot_be_instantiated(self):
        """RetrievalStrategy is abstract and cannot be directly instantiated."""
        with pytest.raises(TypeError):
            RetrievalStrategy()  # type: ignore[abstract]

    def test_required_methods_enforced(self):
        """A subclass missing any abstract method cannot be instantiated."""

        class IncompleteStrategy(RetrievalStrategy):
            @property
            def domain(self):
                return "test"

            def build_index(self, corpus):
                pass

            # search() and compute_families() are missing

        with pytest.raises(TypeError):
            IncompleteStrategy()

    def test_concrete_subclass_ok(self):
        """A fully implemented subclass can be instantiated."""

        class DummyStrategy(RetrievalStrategy):
            @property
            def domain(self):
                return "dummy"

            def build_index(self, corpus):
                pass

            def search(self, query, k=25):
                return RetrievalResult(candidates=[])

            def compute_families(self, candidates):
                return []

        s = DummyStrategy()
        assert s.domain == "dummy"


# =====================================================================
# TestDocumentStrategy  (~8 tests)
# =====================================================================


class TestDocumentStrategy:
    """Document strategy using sentence-transformers + FAISS."""

    def test_build_and_search(self, doc_strategy, document_chunks):
        """Build index and search by text returns candidates."""
        doc_strategy.build_index(document_chunks)
        result = doc_strategy.search({"text": "How do I get a refund?"})

        assert isinstance(result, RetrievalResult)
        assert len(result.candidates) > 0

    def test_topic_filter(self, doc_strategy, document_chunks):
        """allowed_topics restricts results to matching topics."""
        doc_strategy.build_index(document_chunks)
        result = doc_strategy.search(
            {"text": "refund policy", "allowed_topics": ["refund"]}
        )
        for c in result.candidates:
            assert c.metadata["topic"] == "refund"

    def test_doc_type_boost(self, doc_strategy, document_chunks):
        """Doc-type boost affects ranking (at least doesn't crash)."""
        doc_strategy.build_index(document_chunks)
        result = doc_strategy.search(
            {"text": "refund", "allowed_doc_types": ["faq"]}
        )
        # Just verify it completes and returns candidates
        assert isinstance(result, RetrievalResult)

    def test_empty_corpus(self, doc_strategy):
        """Empty corpus returns empty results."""
        doc_strategy.build_index([])
        result = doc_strategy.search({"text": "anything"})
        assert len(result.candidates) == 0

    def test_compute_families_by_topic(self, doc_strategy, document_chunks):
        """compute_families() groups by topic with coherence metrics."""
        doc_strategy.build_index(document_chunks)
        result = doc_strategy.search({"text": "banking loan refund"})
        families = doc_strategy.compute_families(result.candidates)

        assert isinstance(families, list)
        topics_seen = {f.structural_signature.phase for f in families}
        # Should have at least one topic family
        assert len(topics_seen) >= 1

    def test_weak_support_flag(self, doc_strategy):
        """Small groups get uncertainty_flags.weak_support = True."""
        # Build with only 2 chunks in one topic
        chunks = [
            DocumentChunk(chunk_id="a1", text="Tiny topic alpha one", topic="alpha"),
            DocumentChunk(chunk_id="a2", text="Tiny topic alpha two", topic="alpha"),
        ]
        doc_strategy.build_index(chunks)
        result = doc_strategy.search({"text": "alpha"})
        families = doc_strategy.compute_families(result.candidates)

        for f in families:
            if f.coherence.size < 5:
                assert f.uncertainty_flags.weak_support is True

    def test_scores_in_range(self, doc_strategy, document_chunks):
        """All scores in [0, 1]."""
        doc_strategy.build_index(document_chunks)
        result = doc_strategy.search({"text": "open savings account"})
        for c in result.candidates:
            assert 0.0 <= c.score <= 1.0

    def test_save_load(self, doc_strategy, document_chunks, tmp_path):
        """Save / load round-trip preserves search."""
        doc_strategy.build_index(document_chunks)
        result_before = doc_strategy.search({"text": "refund"}, k=5)

        save_dir = str(tmp_path / "doc_idx")
        doc_strategy.save(save_dir)

        loaded = DocumentRetrievalStrategy(DocumentRetrievalConfig(min_similarity=0.0))
        loaded.load(save_dir)
        result_after = loaded.search({"text": "refund"}, k=5)

        assert len(result_after.candidates) > 0
        assert [c.candidate_id for c in result_before.candidates] == [
            c.candidate_id for c in result_after.candidates
        ]
