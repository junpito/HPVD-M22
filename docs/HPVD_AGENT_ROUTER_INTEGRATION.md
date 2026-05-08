# HPVD Integration Guide for Agent Router Team

> This document is written for the team building the Agent Router. It explains how HPVD dispatches queries to different retrieval strategies, the input/output contract for each strategy, and the two integration options the Agent Router team needs to decide between.

**Version:** 1.0-draft | **Audience:** Agent Router team | **Owner:** HPVD team

---

## Table of Contents

1. [What HPVD Does](#1-what-hpvd-does)
2. [Strategy Dispatcher - How Routing Works](#2-strategy-dispatcher---how-routing-works)
3. [Available Strategies and Their Contracts](#3-available-strategies-and-their-contracts)
4. [Integration Decision: Two Options](#4-integration-decision-two-options)
5. [Option A: Agent Router Sends Separate Queries](#5-option-a-agent-router-sends-separate-queries)
6. [Option B: HPVD Auto-Dispatches](#6-option-b-hpvd-auto-dispatches)
7. [Comparison Table](#7-comparison-table)
8. [Request and Response Examples](#8-request-and-response-examples)
9. [Health and Readiness](#9-health-and-readiness)
10. [Decision Required from Agent Router Team](#10-decision-required-from-agent-router-team)

---

## 1. What HPVD Does

HPVD is a **knowledge retrieval engine**. It does not make decisions. It retrieves relevant knowledge objects (policy, product, rule mapping, document schema, documents, time series analogs) from an in-memory index and returns them as candidates.

HPVD supports **three retrieval strategies**, each optimized for a different data type. The Agent Router determines what to ask HPVD, and HPVD returns deterministic results.

```
Agent Router
    |
    v
HPVD  +---- KnowledgeRetrievalStrategy  (rules, policy, product)
      +---- DocumentRetrievalStrategy   (full-text document search)
      +---- FinanceRetrievalStrategy    (time series analog search)
    |
    v
Candidates [{type, data, provenance}]
```

---

## 2. Strategy Dispatcher - How Routing Works

HPVD currently routes each query to **exactly one strategy** based on a single field: `scope.domain`.

```
Query arrives
    |
    v  Read scope.domain from query
    |  Example: "knowledge", "document", "finance"
    |
    v  Resolve alias to canonical domain
    |
    |  +------------------+------------------+-----------------------------+
    |  | Input domain     | Canonical domain | Strategy                    |
    |  +------------------+------------------+-----------------------------+
    |  | "knowledge"      | "knowledge"      | KnowledgeRetrievalStrategy  |
    |  | "finance"        | "finance"        | FinanceRetrievalStrategy    |
    |  | "equity"         | "finance"        | FinanceRetrievalStrategy    |
    |  | "document"       | "document"       | DocumentRetrievalStrategy   |
    |  | "chatbot"        | "document"       | DocumentRetrievalStrategy   |
    |  | "refund"         | "document"       | DocumentRetrievalStrategy   |
    |  | "banking"        | "document"       | DocumentRetrievalStrategy   |
    |  | "loan"           | "document"       | DocumentRetrievalStrategy   |
    |  +------------------+------------------+-----------------------------+
    |
    v  Lookup strategy in registry -> execute search -> return candidates
```

**Important:** if `scope.domain` is missing or unrecognized, HPVD returns an error (400).

---

## 3. Available Strategies and Their Contracts

### 3.1 KnowledgeRetrievalStrategy - domain: `"knowledge"`

Retrieves policy, product, rule mapping, and document schema based on sector and observed fields. Deterministic, no ML model involved.

**When to use:** the Agent Router needs to know which rules/policies apply to the user's data.

| Input field | Type | Required | Description |
|-------------|------|----------|-------------|
| `scope.domain` | `"knowledge"` | yes | Routes to this strategy |
| `sector` | string | yes | e.g. `"banking"` |
| `observed_data` | dict | yes | Field-value pairs from Parser |
| `query_id` | string | yes | Unique query identifier |

**Output:** candidates with `knowledge_type` = `"policy"`, `"product"`, `"rule_mapping"`, or `"document_schema"`. Rule mapping is **always** included.

**Latency:** ~8ms

---

### 3.2 DocumentRetrievalStrategy - domain: `"document"`

Semantic search over document chunks using sentence-transformer embeddings + FAISS.

**When to use:** the Agent Router needs to find relevant documents or text passages.

| Input field | Type | Required | Description |
|-------------|------|----------|-------------|
| `scope.domain` | `"document"` | yes | Routes to this strategy |
| `query_payload.text` | string | yes | Search query text |
| `allowed_topics` | list[str] | no | Filter by topic |
| `allowed_doc_types` | list[str] | no | Filter by document type |
| `query_id` | string | yes | Unique query identifier |

**Output:** candidates with similarity scores, chunk text, topic, and provenance.

**Latency:** ~20-50ms (depends on corpus size)

---

### 3.3 FinanceRetrievalStrategy - domain: `"finance"`

Historical analog search based on trajectory matrices (60 trading days x 45 features). Only relevant for capital markets / OHLCV use cases.

**When to use:** the Agent Router needs historical market analogs. Not used for banking/loan workflows.

| Input field | Type | Required | Description |
|-------------|------|----------|-------------|
| `scope.domain` | `"finance"` | yes | Routes to this strategy |
| `query_payload.trajectory` | 60x45 matrix | yes | Price/feature trajectory |
| `query_payload.dna` | list[float] | yes | Trajectory fingerprint |
| `query_id` | string | yes | Unique query identifier |

**Output:** candidates with confidence scores, family assignments.

**Latency:** ~50ms

---

## 4. Integration Decision: Two Options

The Agent Router team needs to decide how to use HPVD when a single user flow requires results from **more than one strategy** (e.g. both rules AND documents).

```
+----------------------------------------------------------------------+
|                                                                      |
|  Option A: Agent Router sends 2 separate queries to HPVD            |
|            Agent Router merges results                               |
|                                                                      |
|  Option B: HPVD adds auto-dispatch (domain = "auto")                |
|            HPVD runs multiple strategies, returns combined results   |
|                                                                      |
+----------------------------------------------------------------------+
```

---

## 5. Option A: Agent Router Sends Separate Queries

The Agent Router decides which strategies to use and sends one query per strategy. HPVD does not change.

```
Agent Router
    |
    +-- Query 1: POST /query  {scope.domain: "knowledge", sector, observed_data}
    |   +-- HPVD returns: rule_mapping, policy, product candidates
    |
    +-- Query 2: POST /query  {scope.domain: "document", query_payload.text: "..."}
    |   +-- HPVD returns: document chunk candidates
    |
    v
Agent Router merges both results and decides next action
```

**Advantages:**

- No HPVD changes needed - works today.
- Agent Router has full control over which strategies to call and when.
- Agent Router can call strategies conditionally (e.g. only search documents if knowledge retrieval indicates missing docs).
- Each query has a clear, single-domain response that is easy to parse.

**Disadvantages:**

- Agent Router needs to know the available domains and their input formats.
- Two network round-trips instead of one (though they can run in parallel).
- Agent Router is responsible for merging results from different strategies.

---

## 6. Option B: HPVD Auto-Dispatches

HPVD adds a new mode: `scope.domain = "auto"`. HPVD detects which strategies are relevant from the query content and runs them all. HPVD returns a combined response.

```
Agent Router
    |
    +-- Single query: POST /query  {scope.domain: "auto", sector, observed_data, query_payload.text}
        |
        HPVD auto-detects:
        +-- Has observed_data + sector?    -> run KnowledgeRetrievalStrategy
        +-- Has query_payload.text?        -> run DocumentRetrievalStrategy
        +-- Has query_payload.trajectory?  -> run FinanceRetrievalStrategy
        |
        +-- Returns combined response:
            {
              "knowledge": {candidates: [...]},
              "document":  {candidates: [...]},
            }
```

**Advantages:**

- Agent Router sends one request, gets all results.
- Agent Router does not need to know domain names or input formats per strategy.
- Simpler Agent Router implementation.

**Disadvantages:**

- Requires HPVD changes (new dispatcher logic, new response format, new tests).
- Auto-detection can be ambiguous if query has partial fields.
- Response format changes from single-domain to multi-domain - all consumers need to handle this.
- May run unnecessary strategies (e.g. document search when only rules are needed), wasting compute.
- Harder to debug: which strategy produced which result?

---

## 7. Comparison Table

| Aspect | Option A: Separate Queries | Option B: Auto-Dispatch |
|--------|---------------------------|------------------------|
| HPVD changes needed | None | Yes (new dispatcher, response format) |
| Agent Router complexity | Moderate (knows domains) | Low (sends one query) |
| Network round-trips | 2 (can be parallel) | 1 |
| Control over which strategies run | Full (Agent Router decides) | Partial (HPVD auto-detects) |
| Response format | Single-domain (current) | Multi-domain (new) |
| Debugging | Clear (one domain per response) | Harder (mixed response) |
| Unnecessary compute | No (only call what you need) | Possible (auto runs all matching) |
| Timeline | Works today | Needs development + testing |
| Conditional strategy calls | Easy (call document only if needed) | Hard (auto runs everything) |

**HPVD team recommendation:** start with **Option A** for the initial integration. It works today, gives the Agent Router full control, and avoids HPVD changes. Option B can be added later if the Agent Router team finds that managing multiple queries is too complex.

---

## 8. Request and Response Examples

### 8.1 Knowledge Query (rules, policy, product)

**Request:**

```json
{
  "commit_id": "COMMIT_20260430_001",
  "sector": "banking",
  "observed": {
    "loan_amount": 50000000,
    "income": 10000000,
    "date_admission": "2026-04-01"
  },
  "availability": {
    "loan_application_form": true,
    "financial_statement": false
  }
}
```

**Note:** the current REST API (`POST /query`) hardcodes `scope.domain = "knowledge"`. If the Agent Router needs to call other domains via REST, a new endpoint or parameter will be added.

**Response:**

```json
{
  "j14": {
    "domain": "knowledge",
    "candidates": [
      {
        "candidate_id": "policy:POLICY_SME_LOAN_V1",
        "score": 1.2,
        "metadata": {
          "knowledge_type": "policy",
          "sector": "banking",
          "data": {
            "policy_id": "POLICY_SME_LOAN_V1",
            "eligibility_rules": {"min_income": 3000000},
            "required_documents": ["loan_application_form", "identity_document"]
          },
          "provenance": {"source": "bank_internal_policy"}
        }
      },
      {
        "candidate_id": "rule_mapping:MAP-BANKING-V3-RULES",
        "score": 1.0,
        "metadata": {
          "knowledge_type": "rule_mapping",
          "sector": "banking",
          "data": {
            "mapping_id": "MAP-BANKING-V3-RULES",
            "rules": [
              {"rule_id": "V3-001", "fields": ["date_admission", "date_application"]}
            ],
            "gate": {"field": "ep_status", "required_value": "EP_KNOWN"}
          },
          "provenance": {"source": "core_binding_definition"}
        }
      }
    ],
    "diagnostics": {
      "sector": "banking",
      "objects_considered": 12,
      "objects_returned": 3,
      "rule_mapping_forced": true
    }
  },
  "j15": {"accepted": ["..."], "rejected": []},
  "j16": {"families": ["..."], "total_families": 2}
}
```

### 8.2 Document Query (semantic search)

**Request (via internal pipeline, not current REST API):**

```json
{
  "query_id": "REQ_DOC_001",
  "scope": {"domain": "document"},
  "query_payload": {"text": "syarat pengajuan kredit usaha kecil"},
  "allowed_topics": ["loan_application"],
  "allowed_doc_types": ["policy_document"]
}
```

**Response:**

```json
{
  "j14": {
    "domain": "document",
    "candidates": [
      {
        "candidate_id": "chunk_0042",
        "score": 0.87,
        "metadata": {
          "text": "Persyaratan pengajuan kredit usaha kecil meliputi...",
          "topic": "loan_application",
          "doc_type": "policy_document",
          "source_document": "DOC_LOAN_POLICY_2026"
        }
      }
    ],
    "diagnostics": {"chunks_searched": 150, "chunks_returned": 5}
  }
}
```

---

## 9. Health and Readiness

The Agent Router should check HPVD health before routing queries.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "ok",
  "corpus_size": 12,
  "domain": "banking"
}
```

**Production target (planned):**

```json
{
  "status": "ok",
  "corpus_size": 12,
  "domain": "banking",
  "corpus_version": "SNAP_2026W18",
  "loaded_at": "2026-04-30T15:00:00Z",
  "strategies_registered": ["knowledge", "document", "finance"]
}
```

| Health field | Agent Router action |
|-------------|---------------------|
| `status: "ok"` and `corpus_size > 0` | Safe to send queries |
| `status: "ok"` and `corpus_size == 0` | Do not send knowledge queries - corpus empty |
| HPVD unreachable | Retry with backoff, then fail gracefully |

---

## 10. Decision Required from Agent Router Team

Please review and decide:

### Decision 1: Dispatch Model

| Option | Description |
|--------|-------------|
| **A: Separate queries** | Agent Router sends 1 query per domain. Works today. Agent Router controls which strategies to call. |
| **B: Auto-dispatch** | Agent Router sends 1 query with `domain: "auto"`. HPVD runs all matching strategies. Requires HPVD development. |
| **C: Both** | Start with Option A now, add Option B later when the Agent Router team requests it. |

### Decision 2: REST API Scope

The current HPVD REST API only exposes `POST /query` for the knowledge domain. If the Agent Router also needs document retrieval via REST:

| Option | Description |
|--------|-------------|
| **Add domain parameter** | `POST /query` accepts `domain` in the request body. Agent Router specifies domain per request. |
| **Separate endpoints** | `POST /query/knowledge`, `POST /query/document`. Each endpoint has its own request format. |

### Decision 3: Corpus Version Tracking

HPVD plans to include `corpus_version` in every response. The Agent Router team should confirm:

- Will the Agent Router store `corpus_version` per user flow / commit?
- Should HPVD reject queries that specify a pinned `corpus_version` that no longer matches the active version?

---

**Contact:** HPVD team
**Last updated:** 2026-04-30
