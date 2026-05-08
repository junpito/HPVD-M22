"""
HPVD REST API
=============

FastAPI application that exposes the HPVD knowledge retrieval pipeline
over HTTP.  The corpus is loaded from the Knowledge Layer REST API once
at startup and held in memory for the lifetime of the process.

Run::

    uvicorn src.hpvd.api:app --host 127.0.0.1 --port 8000 --reload

Endpoints
---------
POST /query
    Accept a Parser SDK output dict, run the knowledge retrieval, return results.

GET /health
    Liveness + corpus-size check.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hpvd.adapters.strategies.knowledge_strategy import KnowledgeRetrievalStrategy
from hpvd.infra.kl_loader import KLCorpusLoader

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Pydantic request model (Parser SDK output format)
# ---------------------------------------------------------------------------


class HPVDQueryRequest(BaseModel):
    """Incoming query in Parser SDK output format.

    Matches the output of ``banking_parser_sdk.build_hpvd_query()``:

    - ``commit_id`` and ``sector`` are required routing fields added by the caller.
      ``commit_id`` aligns with the Manithy commit boundary identifier (J01).
    - ``observed`` contains the merged field values from all Parser runs.
    - ``availability`` is accepted and passed through but not used by HPVD
      internally; it is consumed by PMR and Knowledge Builder in later stages.
    """

    commit_id: str
    sector: str
    observed: Dict[str, Any] = Field(default_factory=dict)
    availability: Dict[str, bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize KL corpus and build the knowledge index at startup."""
    api_key = os.environ.get("KL_API_KEY", "")
    base_url = os.environ.get("KL_BASE_URL", "https://knowledge-layer-production.up.railway.app")
    domain = os.environ.get("KL_DOMAIN", "banking")

    if not api_key:
        logger.error("KL_API_KEY is not set — corpus will be empty")

    loader = KLCorpusLoader(base_url=base_url, api_key=api_key, domain=domain)
    corpus = loader.load_corpus()

    strategy = KnowledgeRetrievalStrategy()
    strategy.build_index(corpus)

    app.state.strategy = strategy
    app.state.corpus_size = len(corpus)
    app.state.domain = domain

    logger.info(
        "HPVD API ready — %d knowledge objects loaded (domain=%s)",
        len(corpus),
        domain,
    )

    yield

    logger.info("HPVD API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HPVD Knowledge Retrieval API",
    description="REST API for the HPVD knowledge retrieval engine (Manithy v1).",
    version="2.0.0-alpha1",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": f"Missing key: {exc}"})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/query")
async def query(request: HPVDQueryRequest) -> Dict[str, Any]:
    """Run the HPVD knowledge retrieval and return results.

    Accepts Parser SDK output format (``observed`` + ``availability``).
    Directly calls ``KnowledgeRetrievalStrategy.search()`` and
    ``compute_families()`` — no J-file envelope or pipeline engine.

    Returns a dict with keys ``candidates``, ``families``, ``diagnostics``.

    Raises
    ------
    503 if the corpus is empty (KL was unreachable at startup).
    400 if the payload is malformed (ValueError or KeyError).
    500 for unexpected errors.
    """
    if app.state.corpus_size == 0:
        raise HTTPException(
            status_code=503,
            detail="Knowledge corpus is empty — Knowledge Layer was unreachable at startup.",
        )

    # Build query dict for KnowledgeRetrievalStrategy
    query_dict = {
        "query_id": request.commit_id,
        "sector": request.sector,
        "observed_data": request.observed,
    }

    try:
        strategy: KnowledgeRetrievalStrategy = app.state.strategy

        # Retrieve candidates
        result = strategy.search(query_dict)

        # Compute families
        families = strategy.compute_families(result.candidates)

        return {
            "query_id": request.commit_id,
            "candidates": [c.to_dict() for c in result.candidates],
            "families": [f.to_dict() for f in families],
            "diagnostics": result.diagnostics,
        }
    except (ValueError, KeyError):
        raise
    except Exception as exc:
        logger.exception("Unexpected error processing query %s", request.commit_id)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Return service liveness and corpus metadata."""
    return {
        "status": "ok",
        "corpus_size": app.state.corpus_size,
        "domain": app.state.domain,
    }
