"""
Contract Tests for HPVD API Surface
====================================

Verify that the frozen MVP contract (HPVDInputBundle, build_from_bundles,
search_families) behaves correctly under valid and invalid inputs.
"""

import json
import warnings
import pytest
import numpy as np

from src.hpvd import HPVDEngine, HPVDConfig
from src.hpvd.trajectory import HPVDInputBundle, Trajectory
from src.hpvd.embedding import EmbeddingComputer
from src.hpvd.engine import HPVD_Output
from src.hpvd.family import (
    AnalogFamily, FamilyMember, FamilyCoherence,
    StructuralSignature, UncertaintyFlags,
)
from src.hpvd.synthetic_data_generator import SyntheticDataGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_bundle(seed: int = 0) -> HPVDInputBundle:
    """Create a minimal valid HPVDInputBundle."""
    rng = np.random.RandomState(seed)
    return HPVDInputBundle(
        trajectory=rng.randn(60, 45).astype(np.float32),
        dna=rng.randn(16).astype(np.float32),
        geometry_context={"LTV": 0.3, "LVC": 0.1, "K": 5.0},
        metadata={
            "trajectory_id": f"test_{seed:04d}",
            "regime_id": "R1",
            "trajectory_horizon": "60",
            "state_dim": "45",
            "dna_version": "v1",
            "schema_version": "hpvd_input_v1",
            "timestamp": "2020-01-01T00:00:00+00:00",
        },
    )


# ===========================================================================
# 1. HPVDInputBundle.validate() — shape & type checks
# ===========================================================================

class TestBundleValidation:
    """Validate HPVDInputBundle contract enforcement."""

    def test_valid_bundle_passes(self):
        bundle = _make_valid_bundle()
        assert bundle.validate() is True

    def test_trajectory_not_ndarray(self):
        bundle = _make_valid_bundle()
        bundle.trajectory = [[1, 2], [3, 4]]  # list, not ndarray
        with pytest.raises(ValueError, match="trajectory must be np.ndarray"):
            bundle.validate()

    def test_trajectory_wrong_ndim(self):
        bundle = _make_valid_bundle()
        bundle.trajectory = np.zeros(100, dtype=np.float32)  # 1-D
        with pytest.raises(ValueError, match="trajectory must be 2-D"):
            bundle.validate()

    def test_trajectory_zero_dim(self):
        bundle = _make_valid_bundle()
        bundle.trajectory = np.zeros((0, 45), dtype=np.float32)
        with pytest.raises(ValueError, match="dimensions must be > 0"):
            bundle.validate()

    def test_trajectory_contains_nan(self):
        bundle = _make_valid_bundle()
        bundle.trajectory[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            bundle.validate()

    def test_dna_not_ndarray(self):
        bundle = _make_valid_bundle()
        bundle.dna = [1.0, 2.0]
        with pytest.raises(ValueError, match="dna must be np.ndarray"):
            bundle.validate()

    def test_dna_wrong_ndim(self):
        bundle = _make_valid_bundle()
        bundle.dna = np.zeros((4, 4), dtype=np.float32)  # 2-D
        with pytest.raises(ValueError, match="dna must be 1-D"):
            bundle.validate()

    def test_geometry_context_not_dict(self):
        bundle = _make_valid_bundle()
        bundle.geometry_context = "bad"
        with pytest.raises(ValueError, match="geometry_context must be dict"):
            bundle.validate()

    def test_metadata_not_dict(self):
        bundle = _make_valid_bundle()
        bundle.metadata = 42
        with pytest.raises(ValueError, match="metadata must be dict"):
            bundle.validate()

    def test_outcome_leakage_rejected(self):
        """Metadata must not contain outcome fields."""
        bundle = _make_valid_bundle()
        bundle.metadata["label_h1"] = "1"
        with pytest.raises(ValueError, match="outcome fields"):
            bundle.validate()

    def test_multiple_outcome_keys_rejected(self):
        bundle = _make_valid_bundle()
        bundle.metadata["return_h1"] = "0.05"
        bundle.metadata["p_up"] = "0.7"
        with pytest.raises(ValueError, match="outcome fields"):
            bundle.validate()


# ===========================================================================
# 2. EmbeddingComputer lifecycle guard
# ===========================================================================

class TestEmbeddingLifecycle:
    """Guard: transform without fit must fail."""

    def test_transform_before_fit_raises(self):
        ec = EmbeddingComputer(n_components=256)
        matrix = np.random.randn(60, 45).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            ec.transform(matrix)

    def test_transform_batch_before_fit_raises(self):
        ec = EmbeddingComputer(n_components=256)
        matrices = np.random.randn(5, 60, 45).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            ec.transform_batch(matrices)

    def test_fit_then_transform_succeeds(self):
        ec = EmbeddingComputer(n_components=32)
        matrices = np.random.randn(10, 60, 45).astype(np.float32)
        ec.fit(matrices)
        emb = ec.transform(matrices[0])
        assert emb.shape == (32,)


# ===========================================================================
# 3. Deprecation warnings
# ===========================================================================

class TestDeprecationWarnings:
    """Legacy API paths must emit DeprecationWarning."""

    def test_build_emits_deprecation(self):
        engine = HPVDEngine(HPVDConfig())
        traj = Trajectory()  # default valid trajectory
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.build([traj])
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "build_from_bundles" in str(dep_warnings[0].message)

    def test_search_families_trajectory_emits_deprecation(self):
        """Passing a Trajectory (not HPVDInputBundle) should warn."""
        generator = SyntheticDataGenerator(seed=99)
        data = generator.generate_scenario_a(n_historical=5)
        engine = HPVDEngine(HPVDConfig())
        engine.build_from_bundles(data["historical"])

        # Build a legacy Trajectory query
        query_traj = engine._bundle_to_trajectory(data["query"][0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.search_families(query_traj)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "HPVDInputBundle" in str(dep_warnings[0].message)

    def test_search_families_bundle_no_deprecation(self):
        """HPVDInputBundle path must NOT emit DeprecationWarning."""
        generator = SyntheticDataGenerator(seed=99)
        data = generator.generate_scenario_a(n_historical=5)
        engine = HPVDEngine(HPVDConfig())
        engine.build_from_bundles(data["historical"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.search_families(data["query"][0])
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            # build_from_bundles internally calls build(), which now warns.
            # But search_families itself must NOT add one.
            search_dep = [
                x for x in dep_warnings
                if "search_families" in str(x.message) or "HPVDInputBundle" in str(x.message)
            ]
            assert len(search_dep) == 0


# ===========================================================================
# 4. Contract: build_from_bundles + search_families round-trip
# ===========================================================================

class TestContractRoundTrip:
    """End-to-end contract: bundles in → HPVD_Output out."""

    @pytest.fixture
    def engine_with_data(self):
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_scenario_a(n_historical=10)
        engine = HPVDEngine(HPVDConfig())
        engine.build_from_bundles(data["historical"])
        return engine, data

    def test_output_has_required_fields(self, engine_with_data):
        engine, data = engine_with_data
        output = engine.search_families(data["query"][0])

        assert hasattr(output, "analog_families")
        assert hasattr(output, "retrieval_diagnostics")
        assert hasattr(output, "metadata")
        assert output.metadata.get("schema_version") == "hpvd_output_v1"

    def test_output_deterministic(self, engine_with_data):
        engine, data = engine_with_data
        q = data["query"][0]
        out1 = engine.search_families(q)
        out2 = engine.search_families(q)

        ids1 = [f.family_id for f in out1.analog_families]
        ids2 = [f.family_id for f in out2.analog_families]
        assert ids1 == ids2

    def test_invalid_bundle_rejected_at_search(self):
        """search_families must reject an invalid bundle."""
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_scenario_a(n_historical=5)
        engine = HPVDEngine(HPVDConfig())
        engine.build_from_bundles(data["historical"])

        bad_bundle = _make_valid_bundle()
        bad_bundle.trajectory = np.zeros(100, dtype=np.float32)  # wrong shape
        with pytest.raises(ValueError):
            engine.search_families(bad_bundle)


# ===========================================================================
# 5. HPVD_Output serialization (hpvd_output_v1 contract)
# ===========================================================================

class TestHPVDOutputSerializer:
    """Validate to_dict / to_json / from_dict round-trip."""

    @pytest.fixture
    def sample_output(self):
        """Build a real HPVD_Output via the engine."""
        generator = SyntheticDataGenerator(seed=77)
        data = generator.generate_scenario_a(n_historical=10)
        engine = HPVDEngine(HPVDConfig())
        engine.build_from_bundles(data["historical"])
        return engine.search_families(data["query"][0])

    # --- to_dict basic structure ---

    def test_to_dict_has_required_keys(self, sample_output):
        d = sample_output.to_dict()
        assert "metadata" in d
        assert "retrieval_diagnostics" in d
        assert "analog_families" in d

    def test_to_dict_schema_version(self, sample_output):
        d = sample_output.to_dict()
        assert d["metadata"]["schema_version"] == "hpvd_output_v1"

    def test_to_dict_family_structure(self, sample_output):
        d = sample_output.to_dict()
        for af in d["analog_families"]:
            assert "family_id" in af
            assert "members" in af
            assert "coherence" in af
            assert "structural_signature" in af
            assert "uncertainty_flags" in af
            for m in af["members"]:
                assert "trajectory_id" in m
                assert "confidence" in m

    # --- to_json ---

    def test_to_json_is_valid_json(self, sample_output):
        text = sample_output.to_json()
        parsed = json.loads(text)
        assert parsed["metadata"]["schema_version"] == "hpvd_output_v1"

    def test_to_json_with_indent(self, sample_output):
        text = sample_output.to_json(indent=2)
        assert "\n" in text  # pretty-printed
        parsed = json.loads(text)
        assert "analog_families" in parsed

    # --- from_dict round-trip ---

    def test_round_trip_to_dict_from_dict(self, sample_output):
        d = sample_output.to_dict()
        restored = HPVD_Output.from_dict(d)

        # Same number of families
        assert len(restored.analog_families) == len(sample_output.analog_families)

        # Same family IDs
        orig_ids = [f.family_id for f in sample_output.analog_families]
        rest_ids = [f.family_id for f in restored.analog_families]
        assert orig_ids == rest_ids

        # Same member counts per family
        for orig_f, rest_f in zip(sample_output.analog_families, restored.analog_families):
            assert len(orig_f.members) == len(rest_f.members)

        # Metadata identical
        assert restored.metadata == sample_output.metadata

        # Diagnostics identical
        assert restored.retrieval_diagnostics == sample_output.retrieval_diagnostics

    def test_round_trip_via_json(self, sample_output):
        """to_json → json.loads → from_dict must reconstruct identically."""
        text = sample_output.to_json()
        d = json.loads(text)
        restored = HPVD_Output.from_dict(d)

        for orig_f, rest_f in zip(sample_output.analog_families, restored.analog_families):
            assert orig_f.family_id == rest_f.family_id
            assert orig_f.coherence.size == rest_f.coherence.size
            assert abs(orig_f.coherence.mean_confidence - rest_f.coherence.mean_confidence) < 1e-9
            assert orig_f.structural_signature.phase == rest_f.structural_signature.phase
            assert orig_f.uncertainty_flags.weak_support == rest_f.uncertainty_flags.weak_support

    # --- from_dict error cases ---

    def test_from_dict_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing required key"):
            HPVD_Output.from_dict({"metadata": {}, "retrieval_diagnostics": {}})

    def test_from_dict_wrong_schema_raises(self):
        d = {
            "metadata": {"schema_version": "unknown_v99"},
            "retrieval_diagnostics": {},
            "analog_families": [],
        }
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            HPVD_Output.from_dict(d)

    def test_from_dict_empty_families_ok(self):
        d = {
            "metadata": {"schema_version": "hpvd_output_v1"},
            "retrieval_diagnostics": {"families_formed": 0},
            "analog_families": [],
        }
        restored = HPVD_Output.from_dict(d)
        assert len(restored.analog_families) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
