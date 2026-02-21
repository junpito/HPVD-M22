"""
Shared pytest fixtures and configuration for HPVD tests.

Warning hygiene:
    SWIG-generated FAISS bindings emit ``DeprecationWarning`` on import
    (e.g. ``__init__`` usage in the C extension layer).  These are not
    actionable from user code, so we suppress them here to keep the test
    output clean.  Our own deprecation warnings (from ``src.hpvd``) are
    still visible.
"""

import warnings

import pytest


def pytest_configure(config):
    """Register custom markers and suppress noisy third-party warnings."""
    # SWIG / faiss DeprecationWarnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"swigfaiss.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"faiss.*",
    )
