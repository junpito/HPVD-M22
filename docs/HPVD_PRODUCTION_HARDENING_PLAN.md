# HPVD Production Hardening Plan

## Purpose

This document defines the planned production changes for HPVD knowledge loading, corpus versioning, reload behavior, and operational safety. It is intended for review before implementation.

The current HPVD API loads the Knowledge Layer corpus once during FastAPI startup, builds an in-memory `KnowledgeRetrievalStrategy` index, and serves all `/query` requests from that in-memory pipeline. This is fast for queries, but it creates production risks when knowledge is updated while users are actively using HPVD.

## Current Behavior

### Runtime Load Flow

1. HPVD starts the FastAPI app in `src/hpvd/api.py`.
2. `lifespan()` reads:
   - `KL_API_KEY`
   - `KL_BASE_URL`
   - `KL_DOMAIN`
3. `KLCorpusLoader.load_corpus()` fetches documents from KL.
4. The loader selects the highest `version_number` for each KL document.
5. The loader parses each document `raw_text` as JSON.
6. The loader infers `object_type` from root keys:
   - `policy_id` -> `policy`
   - `product_id` -> `product`
   - `mapping_id` -> `rule_mapping`
   - `doc_type` -> `document_schema`
7. HPVD builds an in-memory knowledge index.
8. `/query` uses the existing in-memory pipeline and does not fetch KL again.

### Existing Production Risks

- Knowledge updates in KL are not visible until HPVD restarts.
- Restarting HPVD can disrupt traffic unless deployment is rolling or blue-green.
- Multiple replicas can serve different corpus versions during rollout.
- The loader currently requests documents with a fixed limit.
- The seed script can create duplicate KL documents instead of updating existing logical objects.
- A failed KL load at startup can result in an empty corpus.
- There is no explicit active snapshot or corpus version in the API response or health status.

### Current Memory Model Problems

HPVD currently keeps the active knowledge corpus inside process memory. This is useful for low-latency deterministic retrieval, but it creates several production issues.

- **Stale in-memory corpus:** after HPVD builds the in-memory index, updates in KL are not visible to `/query` until the process is restarted or a reload mechanism is added.
- **Restart-dependent refresh:** restarting works because it reruns startup loading, but restart is an operational workaround rather than a safe production update mechanism.
- **Future replica inconsistency:** this does not apply to the current single-replica deployment, but it becomes a production risk once HPVD is scaled horizontally. Each HPVD process owns its own memory, so replicas can serve different corpus versions if they restart or reload at different times.
- **No memory-level version identity:** the active in-memory pipeline does not currently expose a strong corpus identifier such as `snapshot_id`, `corpus_version`, or `loaded_at` beyond basic corpus size/domain metadata.
- **Memory growth risk:** if the KL corpus grows significantly, startup loading and in-memory indexing can consume more RAM and increase cold-start time.
- **Duplicate object amplification:** if seed operations create duplicate KL documents, HPVD may load duplicate logical objects into memory, increasing RAM use and potentially changing retrieval results.
- **Failure can produce empty memory state:** if KL is unreachable at startup, HPVD can start with an empty corpus and return 503 for `/query`.
- **Reload safety is not yet defined:** without atomic swap, a future reload implementation could accidentally replace a healthy in-memory index with a failed or partially loaded one.

The production goal is not to remove memory usage entirely. The preferred target is to keep query-time retrieval in memory, but make the memory state explicitly versioned, observable, reloadable, and safe to roll back.

## Target Architecture

> HPVD keeps using an in-memory index for query performance. Knowledge updates are controlled, observable, and deterministic. The architecture is organized into three planes.

### System Context (Production Target)

```
[ LOCAL FILES ]
    │  policy.json, product.json, rule_mapping.json, document_schema.json
    ▼
┌─────────────────────────────────────────────────────────┐
│ Knowledge Publish Plane                                 │
│                                                         │
│  Seed/Upsert Tool (scripts/seed_hpvd_knowledge.py)     │
│  1. Validate schema locally                             │
│  2. Infer object_type from root keys                    │
│  3. Check KL: document exists?                          │
│     → YES: upload new version to existing document      │
│     → NO:  create new document + version 1              │
│  4. Publish corpus snapshot                             │
│       │                                                 │
│       ▼                                                 │
│  Knowledge Layer (KL)                                   │
│  ┌──────────┬──────────┬─────────────┐                  │
│  │ Documents│ Versions │  Snapshots  │                  │
│  └──────────┴──────────┴─────────────┘                  │
└─────────────────────────────────────────────────────────┘
    │  published snapshot
    ▼
┌─────────────────────────────────────────────────────────┐
│ HPVD Control Plane                                      │
│                                                         │
│  Trigger: startup OR POST /admin/reload-knowledge       │
│  (admin auth required for reload)                       │
│       │                                                 │
│       ▼                                                 │
│  Reload Pipeline:                                       │
│  1. Fetch documents from KL (paginated)                 │
│  2. Get latest version per document                     │
│  3. Parse raw_text as JSON                              │
│  4. Infer object_type (policy_id/product_id/...)        │
│  5. Validate corpus completeness                        │
│  6. Build new KnowledgeRetrievalStrategy                │
│  7. Build new HPVDPipelineEngine                        │
│       │                                                 │
│       ▼                                                 │
│  Swap Decision:                                         │
│  ┌────────────────────┐  ┌──────────────────────────┐   │
│  │ corpus valid?  YES │→ │ Atomic swap pipeline     │   │
│  │                 NO │→ │ Keep old pipeline active  │   │
│  └────────────────────┘  └──────────────────────────┘   │
│                                                         │
│  Runtime State:                                         │
│  app.state.pipeline       ← active in-memory pipeline   │
│  app.state.corpus_version ← snapshot_id / loaded_at     │
│  app.state.corpus_size    ← number of objects loaded     │
│  app.state.reload_log     ← audit trail                 │
└─────────────────────────────────────────────────────────┘
    │  active pipeline in memory
    ▼
┌─────────────────────────────────────────────────────────┐
│ Rule Completeness Query Plane                           │
│                                                         │
│  User                                                   │
│  → provides data / documents                            │
│       │                                                 │
│       ▼                                                 │
│  Agent Router                                           │
│  → routes input to Parser                               │
│       │                                                 │
│       ▼                                                 │
│  Parser                                                 │
│  → extracts observed_data                               │
│       │                                                 │
│       ▼                                                 │
│  HPVD  POST /query  (service auth)                      │
│  ┌─────────────────────────────────────────────┐        │
│  │ In-Memory Index                             │        │
│  │ sector filter → field match → rule_mapping  │        │
│  └─────────────────────────────────────────────┘        │
│  → candidates + corpus_version + diagnostics            │
│       │                                                 │
│       ▼                                                 │
│  Agent Router checks completeness:                      │
│  ┌────────────────────┐  ┌──────────────────────────┐   │
│  │ all rules pass? YES│→ │ Proceed to Core / PMR    │   │
│  │              NO    │→ │ Ask user for missing data │   │
│  └────────────────────┘  └──────────────────────────┘   │
│       │                        │                        │
│       ▼                        ▼                        │
│  Core Layer / PMR         User updates → loop again     │
└─────────────────────────────────────────────────────────┘
```

**Key specifications:**

| Parameter | Current | Production Target |
|-----------|---------|-------------------|
| Corpus load trigger | Startup only | Startup + admin reload |
| Pipeline swap | N/A (single load) | Atomic swap, fallback on failure |
| Corpus version identity | Not exposed | `corpus_version` in every response |
| Knowledge publish | Create-only seed | Idempotent upsert + validation |
| Auth on `/query` | None | Service key (Agent Router) |
| Auth on `/admin/reload` | N/A | Admin key |
| Pagination | Fixed `limit=50` | Full pagination |
| Object retirement | Not supported | Soft delete via metadata |

---

### Knowledge Publish Flow (Step-by-Step)

```
Local JSON files (data/hpvd_knowledge/*.json)
    │
    ▼  Step 1: Validate schema locally                    ~instant
    │  Check root key exists (policy_id / product_id / mapping_id / doc_type)
    │  Check required fields per object type
    │
    ▼  Step 2: Infer object_type                          ~instant
    │  policy_id   → policy
    │  product_id  → product
    │  mapping_id  → rule_mapping
    │  doc_type    → document_schema
    │
    ▼  Step 3: Check KL for existing document             ~100ms
    │  Search by stable ID + domain
    │  → EXISTS: upload new version to that document
    │  → NOT FOUND: create new document + version 1
    │
    ▼  Step 4: Upload to KL                               ~200ms
    │  POST /documents (create) or POST /versions (update)
    │  raw_text = JSON.stringify(object)
    │
    ▼  Step 5: Publish snapshot (when ready)               ~100ms
    │  All objects for this domain are versioned and pinned
    │
Output: Published corpus snapshot in KL
```

| Condition | Action |
|-----------|--------|
| Invalid local JSON (missing root key) | Reject before upload, log error |
| Object already exists in KL | Upload new version (idempotent) |
| Object removed from local files | Warn operator, do not auto-delete |
| `--dry-run` flag | Print intended actions, no upload |

---

### HPVD Reload Flow (Step-by-Step)

```
Trigger: Server startup  OR  POST /admin/reload-knowledge (admin auth)
    │
    ▼  Step 1: Fetch documents from KL                    ~500ms
    │  GET /documents?domain={domain}
    │  Paginate until all documents retrieved
    │
    ▼  Step 2: Get latest version per document            ~200ms
    │  GET /documents/{id}/versions
    │  Select highest version_number
    │
    ▼  Step 3: Fetch content                              ~200ms
    │  raw_text from version record (preferred)
    │  OR GET /documents/{id}/versions/{v}/content (fallback)
    │
    ▼  Step 4: Parse raw_text as JSON                     ~1ms
    │  json.loads(raw_text) → dict
    │
    ▼  Step 5: Infer object_type                          ~1ms
    │  Same inference as publish flow
    │  Inject object_type + sector into each object
    │
    ▼  Step 6: Validate corpus                            ~1ms
    │  Check non-empty, required types present
    │  Check no duplicate logical objects
    │
    ▼  Step 7: Build pipeline                             ~5ms
    │  KnowledgeRetrievalStrategy().build_index(corpus)
    │  HPVDPipelineEngine().register_strategy(strategy)
    │
    ▼  Step 8: Swap decision                              ~instant
    │  Corpus valid? → atomic swap app.state.pipeline
    │  Corpus invalid? → keep old pipeline, log failure
    │
Output: Active pipeline in memory + reload audit log
         total ~1s (depends on KL network + corpus size)
```

| Condition | Action |
|-----------|--------|
| KL unreachable | Keep old pipeline (or empty on first start), log error |
| Corpus empty | Reject swap, keep old pipeline |
| Object parse fails | Skip object, log warning with document_id |
| Cannot infer object_type | Skip object, log warning |
| Reload already in progress | Reject new reload (409 Conflict) |
| Reload during `/query` traffic | `/query` keeps using old pipeline until swap completes |

---

### Rule Completeness Query Loop (Step-by-Step)

```
User provides data or documents
    │
    ▼  Step 1: Agent Router routes to Parser              ~varies
    │  Select parser based on sector / document type
    │
    ▼  Step 2: Parser extracts observed_data              ~varies
    │  → {"loan_amount": 50000000, "income": 10000000, ...}
    │
    ▼  Step 3: Agent Router calls HPVD                    ~10ms
    │  POST /query
    │  {commit_id, sector, observed_data}
    │  (service auth: X-API-Key header)
    │
    ▼  Step 4: HPVD retrieves from in-memory index        ~8ms
    │  Sector filter → field match → mandatory rule_mapping
    │  → candidates [{type, data, provenance}]
    │  → corpus_version, diagnostics
    │
    ▼  Step 5: Agent Router checks completeness           ~1ms
    │  Compare observed_data against rule_mapping requirements
    │  Check policy eligibility rules
    │  Check document availability
    │
    ▼  Step 6: Decision
    │  ┌──────────────────────────────────────────┐
    │  │ All rules satisfied?                     │
    │  │  YES → proceed to Core Layer / PMR       │
    │  │  NO  → ask user for missing data         │
    │  │        user updates → go back to Step 1  │
    │  └──────────────────────────────────────────┘
    │
Output: Complete observed_data → Core Layer / PMR
        OR: loop back to user for missing information
```

**Example — Agent Router calls HPVD:**

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

**Example — HPVD response (production target):**

```json
{
  "corpus_version": "SNAP_2026W18",
  "corpus_size": 12,
  "j14": {
    "candidates": [
      {
        "knowledge_type": "policy",
        "data": {"policy_id": "POLICY_SME_LOAN_V1", "...": "..."},
        "provenance": {"source": "bank_internal_policy"}
      },
      {
        "knowledge_type": "rule_mapping",
        "data": {"mapping_id": "MAP-BANKING-V3-RULES", "...": "..."},
        "provenance": {"source": "core_binding_definition"}
      }
    ]
  },
  "diagnostics": {
    "sector": "banking",
    "objects_considered": 12,
    "objects_returned": 3,
    "rule_mapping_forced": true,
    "corpus_complete": true
  }
}
```

| Condition | Action |
|-----------|--------|
| HPVD returns 503 (empty corpus) | Agent Router retries, then escalates |
| HPVD returns empty candidates | Check `corpus_size` — if 0, corpus issue; if >0, sector not found |
| HPVD timeout | Agent Router retries with backoff |
| `corpus_version` changed mid-flow | Agent Router logs revalidation event |

---

## Agent Router Integration Requirements

> HPVD operates as the **rule and knowledge assurance layer** after the Agent Router. The Agent Router is responsible for user interaction. HPVD provides deterministic rule/knowledge evidence that the Agent Router turns into user-facing prompts.

### Version Consistency

The Agent Router loop introduces a production requirement: a single user flow must know which corpus version was used for each HPVD check.

| Field | Source | Purpose |
|-------|--------|---------|
| `corpus_version` | HPVD `/query` response | Identify which knowledge set was used |
| `loaded_at` | HPVD `/query` response | Timestamp of last reload |
| `commit_id` | Agent Router request | Track the user flow |
| `corpus_version_at_start` | Agent Router storage | Pin the flow to a corpus version |

The Agent Router should persist `corpus_version` with the `commit_id` or session. This prevents silent behavior changes when KL publishes a new rule set while a user is in the middle of completing missing information.

### Rule Change Policy During Active User Flows

| Situation | Recommended Action |
|-----------|-------------------|
| User flow starts with corpus `v1` | Record `v1` as `corpus_version_at_start` |
| HPVD reloads to `v2` during flow | Agent Router detects version change |
| Option A: accept newer version | Continue with `v2`, log revalidation event |
| Option B: reject version change | Warn operator, continue with logged mismatch |
| Flow completes | Record final `corpus_version` in audit trail |

Default: accept newer version with logged revalidation event.

### Single-Version Memory Limitation

HPVD holds exactly one corpus version in memory at any time. When a reload succeeds, the old version is discarded.

| Aspect | Current | Production Target |
|--------|---------|-------------------|
| Versions in memory | 1 | 1 (initial), multi-version later if needed |
| Version identity | Not tracked | `corpus_version` in every response |
| Pinning per user flow | Not possible | Via Agent Router recording |
| Rollback | Restart with old snapshot | Reload with old `KL_SNAPSHOT_ID` |

### Error Handling in the Completeness Loop

| Error | Agent Router Response |
|-------|-----------------------|
| HPVD returns 503 (empty corpus) | Retry with backoff, then fail flow with operator alert |
| HPVD returns empty candidates but `corpus_size > 0` | Sector not found or no matching rules — proceed with caution or escalate |
| HPVD network timeout | Retry with backoff, configurable max retries |
| HPVD returns `corpus_complete: false` (partial load) | Log warning, decide whether to proceed or wait for full reload |
| HPVD unavailable for extended period | Fail flow, do not silently mark as complete |

### Output Contract (Production Target)

The current HPVD output is retrieval-oriented (J14/J15/J16). For Agent Router integration, the response should also include:

| Field | Type | Description |
|-------|------|-------------|
| `corpus_version` | string | Active snapshot ID or version identifier |
| `corpus_size` | int | Number of knowledge objects in active index |
| `corpus_complete` | bool | Whether the corpus loaded without skipped objects |
| `candidates` | list | Retrieved knowledge objects (existing J14 format) |
| `diagnostics` | dict | Sector, counts, rule_mapping forced flag |

HPVD does not need to compute `missing_fields` or `failed_rules` itself. The Agent Router or a dedicated adapter can derive completeness from the retrieved `rule_mapping` and `observed_data`. This keeps HPVD a pure retrieval engine and avoids coupling rule evaluation logic into HPVD.

## Strategy 1: Atomic Knowledge Reload

### Goal

Allow HPVD to load updated knowledge without restarting the process and without replacing a healthy index with a broken one.

### Planned Changes

- Add an admin-only reload endpoint, for example:
  - `POST /admin/reload-knowledge`
  - `GET /admin/reload-knowledge/status`
- The reload process should:
  1. Fetch corpus from KL.
  2. Validate corpus is non-empty unless explicitly allowed.
  3. Build a new `KnowledgeRetrievalStrategy`.
  4. Register it in a new `HPVDPipelineEngine`.
  5. Swap `app.state.pipeline` only after the new pipeline is fully ready.
  6. Keep the previous pipeline active if reload fails.

### Implementation Notes

- The reload endpoint should reuse the same load/build logic currently embedded in `lifespan()`.
- Extract shared logic into a helper such as `build_pipeline_from_kl()`.
- Store reload metadata in `app.state`, for example:
  - `corpus_size`
  - `domain`
  - `loaded_at`
  - `load_duration_ms`
  - `last_reload_status`
  - `last_reload_error`
  - `snapshot_id` when available

### Concurrency and Thread Safety

- Only one reload should run at a time. A second reload request while the first is still running should be rejected with a clear status (e.g. 409 Conflict) or queued.
- Reload should run in a background task so it does not block the FastAPI event loop. The endpoint should return immediately with a reload job ID or status URL.
- The swap of `app.state.pipeline` must be atomic from the perspective of in-flight `/query` requests. In Python with a single-process async server, a simple reference assignment is safe. If HPVD later runs with multiple workers or threads, a lock or read-write pattern may be needed.
- In-flight `/query` requests that started before the swap should complete using the old pipeline. Requests that arrive after the swap should use the new pipeline.

### Acceptance Criteria

- `/query` continues serving using the old index while reload is running.
- A failed reload does not set `corpus_size` to zero.
- A successful reload updates the active pipeline and health metadata.
- Reload is protected by admin authentication.
- Concurrent reload requests are rejected or serialized.
- Reload does not block `/query` responses.

## Strategy 2: Snapshot or Version Pinning

### Goal

Ensure all HPVD replicas serve the same corpus version, especially during multi-replica production deployment.

### Planned Changes

- Introduce an explicit active corpus selector:
  - `KL_SNAPSHOT_ID`, or
  - `KL_CORPUS_VERSION`, or
  - another stable pin supported by KL.
- Prefer loading from a snapshot manifest instead of always selecting latest document versions.
- Include the active snapshot or version in health/readiness output.

### Implementation Notes

- If KL snapshot APIs are ready, use snapshot loading as the primary production path.
- If snapshot APIs are not ready, define an interim convention:
  - all KL documents must carry a corpus version in metadata, or
  - HPVD loads only documents matching `KL_CORPUS_VERSION`.
- Avoid mixing latest versions from unrelated publish events.

### Acceptance Criteria

- Two HPVD replicas with the same config load the same corpus.
- Health output identifies the active corpus version.
- Rollback can be performed by pointing HPVD back to a previous snapshot/version.

## Strategy 3: Idempotent Knowledge Upsert

### Goal

Prevent duplicate logical knowledge objects in KL when the seed script is run multiple times.

### Planned Changes

- Change the seed workflow from create-only to idempotent upsert.
- Use stable object IDs:
  - `policy_id`
  - `product_id`
  - `mapping_id`
  - `doc_type`
- Before creating a new KL document, search/list existing documents for the same stable ID and domain.
- If a document exists, upload a new version to that document.
- If it does not exist, create a new document and upload version 1.

### Implementation Notes

- The current `scripts/seed_hpvd_knowledge.py` infers object type and creates a new KL document for each object.
- Add a metadata field or title convention that can be queried reliably.
- Prefer metadata over title parsing if KL supports metadata search.
- Preserve `--dry-run`.
- Add output that clearly states:
  - created document
  - updated existing document
  - skipped invalid object

### Knowledge Object Deletion and Retirement

The current plan covers creating and updating knowledge objects, but does not address removal. In production, a knowledge object may need to be retired (e.g. a policy is replaced, a rule mapping is deprecated).

Options:

- **Soft delete via metadata:** mark the KL document as retired or inactive. The loader skips objects with a retirement flag. This is the safest approach because the object remains in KL for audit.
- **Hard delete from KL:** remove the KL document entirely. Simpler but loses history. Only appropriate if KL has its own version history or backup.
- **Corpus diffing on seed:** the seed script compares local JSON files against KL documents. Objects present in KL but absent from local files can be flagged for review or retirement.

Recommended default: soft delete via metadata. The seed script should warn when local files no longer contain an object that exists in KL, but should not auto-delete without explicit confirmation.

### Acceptance Criteria

- Running the same seed command twice does not create duplicate logical objects.
- Updating `banking_rule_mapping_v3.json` produces a new version of the existing rule mapping document.
- Dry run reports intended create/update behavior.
- Retired or removed objects are handled explicitly, not silently left in KL.

## Strategy 4: Pagination and Corpus Scale

### Goal

Make corpus loading correct when KL contains more documents than a single page.

### Planned Changes

- Replace fixed `limit=50` behavior with pagination.
- Continue fetching until all pages are loaded.
- Add limits and safeguards for production:
  - maximum corpus object count
  - maximum document size
  - timeout per request
  - total reload timeout

### Implementation Notes

- Update `KLCorpusLoader._fetch_documents()` to support KL pagination shape.
- If KL pagination format is not stable, support common response forms:
  - `documents`
  - `data`
  - `next_cursor`
  - `next_page`
  - `total`
- Log page count and total documents loaded.

### Acceptance Criteria

- Loader can fetch more than 50 knowledge documents.
- Loader reports how many pages and documents were fetched.
- Reload fails clearly if corpus exceeds configured safe limits.

## Strategy 5: Schema Validation

### Goal

Fail early on invalid knowledge objects instead of silently producing incomplete retrieval behavior.

### Planned Changes

- Add validation before upload in `scripts/seed_hpvd_knowledge.py`.
- Add validation during HPVD load in `KLCorpusLoader`.
- Validate at least:
  - recognized root ID key exists
  - required fields for each object type exist
  - `sector` or `domain` is consistent
  - rule mappings have valid `rules`
  - rule mappings have valid `gate`

### Implementation Notes

- Reuse existing knowledge schema code where possible.
- Keep validation deterministic and explicit.
- Decide whether invalid KL objects should:
  - fail the entire reload in production, or
  - be skipped with a hard warning in development.

### Acceptance Criteria

- Invalid local JSON is caught before upload.
- Invalid KL objects are reported with document ID and reason.
- Production reload does not silently activate a partially invalid corpus unless explicitly configured.

## Strategy 6: Health, Readiness, and Observability

### Goal

Make corpus state visible to operators and deployment tooling.

### Planned Changes

- Extend `/health` or add `/ready` with:
  - service status
  - `corpus_size`
  - `domain`
  - `snapshot_id` or corpus version
  - `loaded_at`
  - `load_duration_ms`
  - object counts by type
  - last reload status
  - last reload error
- Add structured logs around:
  - startup load
  - reload start
  - reload success
  - reload failure
  - corpus validation failure

### Acceptance Criteria

- Operators can tell which corpus version is active.
- Deployment readiness fails if no valid corpus is loaded.
- Reload failures are visible without inspecting application internals.

## Strategy 7: Production Deployment Model

### Goal

Avoid downtime and inconsistent traffic during updates.

### Recommended Model

- Run at least two HPVD replicas.
- Use readiness checks so a replica only receives traffic after corpus load succeeds.
- Use rolling deployment for normal releases.
- Use blue-green deployment for high-risk corpus or code changes.
- Trigger reload in all replicas only after a KL corpus version is published.

### Graceful Shutdown and In-Flight Requests

- When HPVD receives a shutdown signal (SIGTERM), it should stop accepting new connections but allow in-flight `/query` requests to complete before exiting.
- The current `lifespan()` logs shutdown but does not explicitly drain requests. Uvicorn supports graceful shutdown with a configurable timeout (`--timeout-graceful-shutdown`). This should be set to a reasonable value (e.g. 30 seconds) in the deployment configuration.
- During a rolling deployment, the load balancer should mark the old replica as draining before routing traffic to the new replica.
- If a reload is in progress when shutdown is received, the reload should be cancelled cleanly and the process should exit with the last healthy pipeline state.

### Acceptance Criteria

- Restarting one replica does not interrupt service.
- A bad corpus publish can be rolled back.
- Query traffic is not routed to a replica with empty or failed corpus.
- In-flight requests complete before process exit during graceful shutdown.
- Graceful shutdown timeout is configurable.

## Strategy 8: Security and Access Control

### Goal

Prevent unauthorized access to both admin operations and query endpoints.

### Current State

HPVD currently has no authentication on any endpoint. Both `/query` and `/health` are open. The only authentication in the codebase is outbound: `KLCorpusLoader` sends `X-API-Key` to KL. This is acceptable for development but not for production.

### Planned Changes

**Admin endpoints (reload, status):**

- Protect with a dedicated admin token or key, separate from the KL API key.
- Simplest initial approach: `HPVD_ADMIN_KEY` environment variable, checked via a FastAPI `Depends()` dependency that reads `X-Admin-Key` or `Authorization: Bearer ...` header.
- Rate limit reload endpoints to prevent abuse.
- Log reload actor, timestamp, old corpus version, and new corpus version.

**Query endpoint (`/query`):**

- In the current architecture, HPVD is called by the Agent Router, not directly by end users. This is a service-to-service call.
- Protect with a service token: `HPVD_SERVICE_KEY` environment variable, checked via `X-API-Key` header or similar.
- If HPVD is deployed behind a gateway or service mesh that handles auth, HPVD can trust the gateway and skip its own token check. This should be an explicit configuration choice, not an accidental omission.

**Health endpoint (`/health`):**

- `/health` can remain open for liveness probes.
- If `/health` exposes sensitive metadata (e.g. corpus version, object counts), consider a separate `/ready` endpoint behind auth for detailed status, while keeping `/health` minimal and open.

### Acceptance Criteria

- Public clients cannot call reload endpoints.
- `/query` is not accessible without a valid service credential unless explicitly configured to trust a gateway.
- Every reload attempt has an audit trail.
- Failed authorization is logged without exposing secrets.
- Auth mechanism is configurable via environment variables.

## Suggested Implementation Phases

### Phase 1: Safe Runtime Reload

Files likely involved:

- `src/hpvd/api.py`
- `src/hpvd/kl_loader.py`
- tests for API reload behavior

Deliverables:

- Shared pipeline builder helper.
- Admin reload endpoint.
- Atomic pipeline swap.
- Health metadata.
- Old pipeline remains active on reload failure.

### Phase 2: Loader Correctness

Files likely involved:

- `src/hpvd/kl_loader.py`
- tests for pagination and version selection

Deliverables:

- Pagination support.
- Better corpus load diagnostics.
- Configurable load limits and timeouts.

### Phase 3: Idempotent Knowledge Publishing

Files likely involved:

- `scripts/seed_hpvd_knowledge.py`
- `src/hpvd/adapters/kl_client.py`
- docs for knowledge publishing

Deliverables:

- Upsert behavior.
- Dry-run create/update reporting.
- Duplicate prevention.

### Phase 4: Snapshot Pinning

Files likely involved:

- `src/hpvd/kl_loader.py`
- `src/hpvd/api.py`
- `src/hpvd/adapters/kl_client.py`
- deployment configuration

Deliverables:

- Load by configured snapshot/version.
- Health reports active corpus pin.
- Rollback path documented.

### Phase 5: Validation and Production Documentation

Files likely involved:

- `src/hpvd/adapters/knowledge_schemas.py`
- `scripts/seed_hpvd_knowledge.py`
- `docs/HPVD_REST_API.md`
- new or updated production runbook

Deliverables:

- Pre-upload validation.
- Load-time validation.
- Production runbook for publish, reload, verify, rollback.

### Phase 6: Agent Router Contract

Files likely involved:

- `src/hpvd/api.py`
- `src/hpvd/adapters/pipeline_engine.py`
- `src/hpvd/adapters/strategies/knowledge_strategy.py`
- integration tests for Agent Router requests
- documentation for the HPVD-to-Agent-Router contract

Deliverables:

- HPVD responses include active corpus identity.
- Query contract supports `commit_id`, `sector`, `observed_data`, and optional pinned corpus identity.
- Agent Router can determine missing fields or required user updates from HPVD output or a dedicated adapter.
- Rule completeness loop is documented with request and response examples.
- Revalidation behavior is defined when an active user flow moves to a newer corpus version.

## Testing Plan

### Unit Tests

- Loader parses valid policy/product/rule mapping/schema objects.
- Loader rejects invalid raw text.
- Loader fetches all pages.
- Loader preserves old corpus when reload fails.
- Seed script detects create vs update.
- Corpus identity is attached to pipeline/query metadata.
- Rule completeness adapter maps retrieved rule mappings to missing/satisfied/failed rule state.

### API Tests

- Startup with valid KL corpus returns healthy status.
- Startup with empty corpus returns not-ready status.
- `/query` continues working during reload.
- Reload success updates `corpus_size` and metadata.
- Reload failure keeps old pipeline.
- Admin reload endpoint rejects missing or invalid credentials.
- `/query` returns the corpus identity used for the result.
- `/query` accepts a pinned corpus identity or returns a clear unsupported-version response if pinning is not yet implemented.
- Agent Router-style requests can repeat after user updates and receive deterministic completeness output.

### Operational Tests

- Run two HPVD replicas with the same snapshot and confirm identical corpus metadata.
- Publish new corpus version and reload both replicas.
- Roll back to previous corpus version.
- Simulate KL outage during reload and confirm query traffic still works.
- Start a user flow under one corpus version, publish a newer corpus, and verify the flow either stays pinned or explicitly revalidates.
- Confirm new user flows can use the newer corpus after reload while old flows remain auditable.

### Load and Stress Tests

- Measure `/query` latency and throughput under sustained concurrent requests with the expected production corpus size.
- Measure memory usage after loading the expected maximum corpus size.
- Measure startup time (cold start) with increasing corpus sizes to identify scaling limits.
- Simulate reload under load: trigger a reload while `/query` is receiving traffic, and confirm that query latency does not spike and no requests fail.
- Simulate concurrent reload requests and confirm that only one reload runs at a time.
- Measure memory usage before and after reload to confirm the old pipeline is released.

## Open Decisions for Review

- Should production reload fail the whole corpus if one object is invalid, or skip invalid objects with warnings?
- Should HPVD support reload from latest KL documents during development while requiring snapshot pinning in production?
- What admin authentication mechanism should be used for reload endpoints?
- What is the expected maximum number of knowledge objects per domain?
- Should corpus version be owned by KL snapshots or by HPVD deployment config?
- Should active Agent Router flows be pinned to the starting corpus version until completion?
- Should HPVD itself compute `missing_fields` and `failed_rules`, or should it return retrieved knowledge and let the Agent Router compute completeness?
- What should happen if the Agent Router asks for a corpus version that HPVD no longer has in memory?

## Recommended Defaults

- Production should require snapshot/version pinning.
- Production reload should fail if any required knowledge object is invalid.
- Development can allow latest-version loading for convenience.
- Reload should always preserve the last healthy index on failure.
- Seed should be idempotent by default, with explicit create-only behavior only for tests or demos.
- Agent Router flows should be pinned to the corpus version used at flow start.
- HPVD should include corpus identity in every `/query` response.
- HPVD should expose rule completeness through a dedicated adapter if the Agent Router needs direct `missing_fields` or `required_user_updates`.

## Review Checklist

- Confirm the target production behavior is acceptable.
- Confirm whether snapshot APIs are available in KL.
- Confirm the admin authentication approach.
- Confirm whether idempotent seed should update documents by metadata or title convention.
- Confirm whether implementation should start with Phase 1 only or include Phase 1 and Phase 2 together.
- Confirm whether Agent Router needs HPVD to compute completeness directly or only return rule evidence.
- Confirm the corpus version policy for active user flows.
