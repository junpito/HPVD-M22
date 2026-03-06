"""
Tests for the HPVD Adapter Layer
=================================

Covers:
    - RetrievalStrategy ABC enforcement
    - FinanceRetrievalStrategy (wrapping HPVDEngine)
    - DocumentRetrievalStrategy (sentence-transformers + FAISS)
    - StrategyDispatcher
    - J-File schemas (J13–J16)
    - HPVDPipelineEngine end-to-end

Total: ~33 tests
"""

import json
import uuid

import numpy as np
import pytest

from src.hpvd import HPVDConfig
from src.hpvd.synthetic_data_generator import SyntheticDataGenerator
from src.hpvd.family import FamilyCoherence, StructuralSignature, UncertaintyFlags

from src.hpvd.adapters.retrieval_strategy import (
    FamilyAssignment,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalStrategy,
)
from src.hpvd.adapters.j_file_schemas import (
    J13_PostCoreQuery,
    J14_RetrievalRaw,
    J15_PhaseFilteredSet,
    J16_AnalogFamilyAssignment,
)
from src.hpvd.adapters.strategy_dispatcher import StrategyDispatcher, DOMAIN_ALIASES
from src.hpvd.adapters.j13_adapter import J13Adapter
from src.hpvd.adapters.j14_emitter import J14Emitter
from src.hpvd.adapters.j15_emitter import J15Emitter
from src.hpvd.adapters.j16_emitter import J16Emitter
from src.hpvd.adapters.strategies.finance_strategy import FinanceRetrievalStrategy
from src.hpvd.adapters.strategies.document_strategy import (
    DocumentChunk,
    DocumentRetrievalConfig,
    DocumentRetrievalStrategy,
)
from src.hpvd.adapters.pipeline_engine import HPVDPipelineEngine, PipelineOutput


# =====================================================================
# Helpers / fixtures
# =====================================================================

@pytest.fixture
def generator():
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def finance_strategy():
    config = HPVDConfig(
        default_k=25,
        enable_sparse_filter=True,
        enable_reranking=True,
    )
    return FinanceRetrievalStrategy(config)


@pytest.fixture
def finance_data(generator):
    """Pre-generated scenario-A data for finance tests."""
    return generator.generate_scenario_a(n_historical=20, regime_id="R1")


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
# TestFinanceStrategy  (~6 tests)
# =====================================================================


class TestFinanceStrategy:
    """Finance strategy wrapping HPVDEngine."""

    def test_build_and_search(self, finance_strategy, finance_data):
        """Build from bundles → search → returns RetrievalResult with candidates."""
        finance_strategy.build_index(finance_data["historical"])
        query_bundle = finance_data["query"][0]
        result = finance_strategy.search(
            {"hpvd_input_bundle": query_bundle}, k=25
        )

        assert isinstance(result, RetrievalResult)
        assert len(result.candidates) > 0
        assert result.query_id  # non-empty

    def test_candidate_scores_in_range(self, finance_strategy, finance_data):
        """All candidate scores must be in [0, 1]."""
        finance_strategy.build_index(finance_data["historical"])
        result = finance_strategy.search(
            {"hpvd_input_bundle": finance_data["query"][0]}, k=25
        )
        for c in result.candidates:
            assert 0.0 <= c.score <= 1.0, f"score {c.score} out of [0,1]"

    def test_compute_families(self, finance_strategy, finance_data):
        """compute_families() returns valid FamilyAssignment list."""
        finance_strategy.build_index(finance_data["historical"])
        result = finance_strategy.search(
            {"hpvd_input_bundle": finance_data["query"][0]}, k=25
        )
        families = finance_strategy.compute_families(result.candidates)

        assert isinstance(families, list)
        assert len(families) > 0
        for f in families:
            assert isinstance(f, FamilyAssignment)
            assert f.coherence.size > 0

    def test_round_trip_j14_j15_j16(self, finance_strategy, finance_data):
        """Finance output can be emitted as J14 / J15 / J16."""
        finance_strategy.build_index(finance_data["historical"])
        result = finance_strategy.search(
            {"hpvd_input_bundle": finance_data["query"][0]}, k=25
        )
        families = finance_strategy.compute_families(result.candidates)

        j14 = J14Emitter.emit("q1", "finance", result)
        j15 = J15Emitter.emit("q1", result)
        j16 = J16Emitter.emit("q1", families)

        assert j14.query_id == "q1"
        assert j15.query_id == "q1"
        assert j16.query_id == "q1"
        assert len(j14.candidates) > 0
        assert j16.total_families > 0

    def test_deterministic_replay(self, finance_data):
        """Same query on same engine → identical results."""
        config = HPVDConfig(default_k=25, enable_sparse_filter=True, enable_reranking=True)
        s1 = FinanceRetrievalStrategy(config)
        s1.build_index(finance_data["historical"])
        r1 = s1.search({"hpvd_input_bundle": finance_data["query"][0]}, k=10)

        s2 = FinanceRetrievalStrategy(config)
        s2.build_index(finance_data["historical"])
        r2 = s2.search({"hpvd_input_bundle": finance_data["query"][0]}, k=10)

        ids1 = [c.candidate_id for c in r1.candidates]
        ids2 = [c.candidate_id for c in r2.candidates]
        assert ids1 == ids2

    def test_save_load(self, finance_strategy, finance_data, tmp_path):
        """Save → load preserves search capability."""
        finance_strategy.build_index(finance_data["historical"])
        result_before = finance_strategy.search(
            {"hpvd_input_bundle": finance_data["query"][0]}, k=10
        )

        save_dir = str(tmp_path / "finance_idx")
        finance_strategy.save(save_dir)

        loaded = FinanceRetrievalStrategy()
        loaded.load(save_dir)
        result_after = loaded.search(
            {"hpvd_input_bundle": finance_data["query"][0]}, k=10
        )

        assert len(result_after.candidates) > 0
        assert [c.candidate_id for c in result_before.candidates] == [
            c.candidate_id for c in result_after.candidates
        ]


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


# =====================================================================
# TestStrategyDispatcher  (~4 tests)
# =====================================================================


class TestStrategyDispatcher:
    """Strategy dispatcher routing."""

    def test_register_and_dispatch_finance(self, finance_strategy):
        d = StrategyDispatcher()
        d.register(finance_strategy)
        j13 = J13_PostCoreQuery(query_id="q1", scope={"domain": "finance"})
        assert d.dispatch(j13).domain == "finance"

    def test_dispatch_via_alias(self, doc_strategy):
        d = StrategyDispatcher()
        d.register(doc_strategy)
        j13 = J13_PostCoreQuery(query_id="q2", scope={"domain": "chatbot"})
        assert d.dispatch(j13).domain == "document"

    def test_unregistered_domain_raises(self):
        d = StrategyDispatcher()
        j13 = J13_PostCoreQuery(query_id="q3", scope={"domain": "unknown"})
        with pytest.raises(ValueError, match="No strategy registered"):
            d.dispatch(j13)

    def test_multiple_strategies(self, finance_strategy, doc_strategy):
        d = StrategyDispatcher()
        d.register(finance_strategy)
        d.register(doc_strategy)
        assert len(d.registered_domains) == 2
        j_fin = J13_PostCoreQuery(query_id="q4", scope={"domain": "equity"})
        j_doc = J13_PostCoreQuery(query_id="q5", scope={"domain": "banking"})
        assert d.dispatch(j_fin).domain == "finance"
        assert d.dispatch(j_doc).domain == "document"


# =====================================================================
# TestJFileSchemas  (~6 tests)
# =====================================================================


class TestJFileSchemas:
    """J13–J16 round-trip and validation."""

    def test_j13_round_trip(self):
        j = J13_PostCoreQuery(
            query_id="q1",
            scope={"domain": "finance", "action_class": "analog_search"},
            allowed_topics=["R1"],
            allowed_corpora=["equity_us"],
            query_payload={"text": "hello"},
        )
        d = j.to_dict()
        j2 = J13_PostCoreQuery.from_dict(d)
        assert j2.query_id == j.query_id
        assert j2.scope == j.scope
        assert j2.allowed_topics == j.allowed_topics
        assert j2.query_payload == j.query_payload

    def test_j14_round_trip(self):
        j = J14_RetrievalRaw(query_id="q1", domain="finance", candidates=[{"id": "c1"}])
        d = j.to_dict()
        j2 = J14_RetrievalRaw.from_dict(d)
        assert j2.query_id == "q1"
        assert j2.domain == "finance"
        assert j2.candidates == [{"id": "c1"}]

    def test_j15_round_trip(self):
        j = J15_PhaseFilteredSet(
            query_id="q1",
            accepted=[{"id": "a"}],
            rejected=[{"id": "r"}],
            filter_criteria={"topic": "R1"},
        )
        d = j.to_dict()
        j2 = J15_PhaseFilteredSet.from_dict(d)
        assert j2.accepted == j.accepted
        assert j2.rejected == j.rejected

    def test_j16_round_trip(self):
        j = J16_AnalogFamilyAssignment(
            query_id="q1",
            families=[{"family_id": "AF_001"}],
            total_members=5,
            total_families=1,
            metadata={"domain": "finance"},
        )
        d = j.to_dict()
        j2 = J16_AnalogFamilyAssignment.from_dict(d)
        assert j2.total_families == 1
        assert j2.families == j.families

    def test_j13_missing_required_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            J13_PostCoreQuery.from_dict({"scope": {"domain": "finance"}})

    def test_schema_ids_correct(self):
        assert J13_PostCoreQuery(query_id="x", scope={}).schema_id == "manithy.post_core_query.v2"
        assert J14_RetrievalRaw(query_id="x", domain="d").schema_id == "manithy.hpvd_retrieval_raw.v1"
        assert J15_PhaseFilteredSet(query_id="x").schema_id == "manithy.phase_filtered_set.v1"
        assert J16_AnalogFamilyAssignment(query_id="x").schema_id == "manithy.analog_family_assignment.v1"


# =====================================================================
# TestPipelineEngine  (~6 tests)
# =====================================================================


class TestPipelineEngine:
    """End-to-end pipeline tests."""

    def _make_finance_j13(self, bundle) -> dict:
        """Build a J13 dict for finance domain."""
        return {
            "query_id": "fin_q1",
            "scope": {"domain": "finance", "action_class": "analog_search"},
            "allowed_topics": [],
            "query_payload": {"hpvd_input_bundle": bundle},
        }

    def _make_document_j13(self, text: str = "refund policy") -> dict:
        return {
            "query_id": "doc_q1",
            "scope": {"domain": "chatbot"},
            "allowed_topics": [],
            "query_payload": {"text": text},
        }

    def test_finance_end_to_end(self, finance_strategy, finance_data):
        """J13 finance → J14+J15+J16 with valid structure."""
        finance_strategy.build_index(finance_data["historical"])
        pipeline = HPVDPipelineEngine(strategies=[finance_strategy])

        j13_dict = self._make_finance_j13(finance_data["query"][0])
        out = pipeline.process_query(j13_dict, k=25)

        assert isinstance(out, PipelineOutput)
        assert out.j14.domain == "finance"
        assert len(out.j14.candidates) > 0
        assert out.j16.total_families > 0

    def test_document_end_to_end(self, doc_strategy, document_chunks):
        """J13 chatbot → J14+J15+J16 with valid structure."""
        doc_strategy.build_index(document_chunks)
        pipeline = HPVDPipelineEngine(strategies=[doc_strategy])

        j13_dict = self._make_document_j13("How to get a refund?")
        out = pipeline.process_query(j13_dict, k=10)

        assert isinstance(out, PipelineOutput)
        assert out.j14.domain == "document"
        assert len(out.j14.candidates) > 0
        assert out.j16.total_families >= 1

    def test_unknown_domain_raises(self):
        pipeline = HPVDPipelineEngine()
        j13_dict = {
            "query_id": "bad",
            "scope": {"domain": "alien"},
            "query_payload": {},
        }
        with pytest.raises(ValueError, match="No strategy registered"):
            pipeline.process_query(j13_dict)

    def test_j16_families_have_coherence_and_uncertainty(
        self, finance_strategy, finance_data
    ):
        """J16 families should contain coherence & uncertainty_flags."""
        finance_strategy.build_index(finance_data["historical"])
        pipeline = HPVDPipelineEngine(strategies=[finance_strategy])

        j13_dict = self._make_finance_j13(finance_data["query"][0])
        out = pipeline.process_query(j13_dict, k=25)

        for fam_dict in out.j16.families:
            assert "coherence" in fam_dict
            assert "uncertainty_flags" in fam_dict
            assert "mean_confidence" in fam_dict["coherence"]

    def test_pipeline_output_to_dict_json_safe(
        self, finance_strategy, finance_data
    ):
        """PipelineOutput.to_dict() produces a JSON-serializable dict."""
        finance_strategy.build_index(finance_data["historical"])
        pipeline = HPVDPipelineEngine(strategies=[finance_strategy])

        j13_dict = self._make_finance_j13(finance_data["query"][0])
        out = pipeline.process_query(j13_dict, k=25)

        d = out.to_dict()
        # Must not raise
        serialized = json.dumps(d, ensure_ascii=False)
        assert isinstance(serialized, str)

    def test_multiple_domains_in_same_pipeline(
        self, finance_strategy, finance_data, doc_strategy, document_chunks
    ):
        """Both domains registered and dispatched correctly."""
        finance_strategy.build_index(finance_data["historical"])
        doc_strategy.build_index(document_chunks)
        pipeline = HPVDPipelineEngine(
            strategies=[finance_strategy, doc_strategy]
        )

        out_fin = pipeline.process_query(
            self._make_finance_j13(finance_data["query"][0]), k=10
        )
        out_doc = pipeline.process_query(
            self._make_document_j13("open savings account"), k=10
        )

        assert out_fin.j14.domain == "finance"
        assert out_doc.j14.domain == "document"
