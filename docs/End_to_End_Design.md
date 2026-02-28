# Manithy — End-to-End Design

> Dokumen ini mendeskripsikan arsitektur end-to-end sistem **Manithy**, mulai dari input user hingga output penjelasan, untuk tiga domain: **Finance**, **Chatbot (Refund)**, dan **Banking (Loan)**.

---

## Daftar Isi

1. [Business Flow](#1-business-flow)
2. [Tech Flow](#2-tech-flow)
3. [J-Files Reference](#3-j-files-reference)
4. [Format VectorState](#4-format-vectorstate)
5. [Task Assignment](#5-task-assignment)

---

## 1. Business Flow

Pipeline bisnis terdiri dari **18 stage** yang mentransformasi input mentah menjadi penjelasan berbasis reasoning terstruktur.

### Stage 1.0 — UPLOAD / INPUT

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | User memilih: Asset, Time range, Granularity (range of each row). Tidak ada file upload. Input dikirim ke adapter finance. **Hasil:** Permintaan trade dengan parameter lengkap. Stage berikutnya consume: Parameter + pilihan asset + waktu. |
| **Chatbot** | User memulai percakapan: Mengajukan refund, Memberikan order ID, Memberikan alasan. Chatbot bertanya hanya yang diperlukan: Nomor order, Alasan refund, Konfirmasi detail pembayaran. Adapter chatbot juga mengambil data SDK (payment state, chargeback state, dll). **Hasil:** Data refund lengkap siap diproses. |
| **Banking** | User mengisi formulir pengajuan pinjaman, upload data diri, pilih jumlah pinjaman, pilih region. Adapter banking: Ekstraksi data dari dokumen, validasi format, hitung debt ratio, cek kelengkapan dokumen. **Hasil:** Data aplikasi pinjaman lengkap. Stage berikutnya consume: Form input + hasil ekstraksi dokumen. |

### Stage 2.0 — IMMUTABLE SNAPSHOT

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Adapter mengambil snapshot market: Market price saat itu, Volatilitas, Struktur market, Semua sinyal relevan. **Hasil:** Market state terkunci pada satu titik waktu. |
| **Chatbot** | Sistem membekukan: State order saat itu, Status pembayaran, Status chargeback, Detail transaksi. **Hasil:** Snapshot refund pada satu titik waktu. |
| **Banking** | Sistem membekukan: Nilai pengajuan, Data income, Debt ratio saat itu, Status dokumen, Data kredit bureau (jika ada). **Hasil:** Snapshot aplikasi pinjaman. |

### Stage 3.0 — CASE METADATA CAPTURE

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Sistem mencatat: Siapa tenantnya, Jenis aksi (`TRADE_EXECUTION`), Commit boundary. **Hasil:** Identitas kasus dan konteks resmi. |
| **Chatbot** | Sistem mencatat: Tenant (misalnya `MERCHANT_EU`), Action class (`CHATBOT_EXECUTION`), Boundary (`REFUND_COMMIT`). **Hasil:** Kasus refund resmi teridentifikasi. |
| **Banking** | Sistem mencatat: Tenant (misalnya `BANKING_CORE`), Action class (`LOAN_EXECUTION`), Commit boundary (`LOAN_SUBMISSION`). **Hasil:** Kasus pinjaman resmi teridentifikasi. |

### Stage 4.0 — STRUCTURED CASE JSON

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Semua data mentah dari snapshot disusun dalam format terstruktur dan konsisten. Belum ada evaluasi. **Hasil:** Kasus yang sudah rapi dan siap diproses secara deterministik. |
| **Chatbot** | Semua data disusun secara rapi: Amount, Currency, Region, Refund mode, Availability flags. **Hasil:** Kasus refund terstruktur. |
| **Banking** | Semua data disusun secara terstruktur: Requested amount, Income, Debt ratio, Collateral, Availability flags. **Hasil:** Kasus pinjaman dalam format rapi. |

### Stage 5.0 — DETERMINISTIC PROJECTION (Merge Metadata + Doc)

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Sistem membangun representasi keadaan trade: Sinyal volatilitas, Rasio risiko, Struktur market, Ketersediaan data. Semua dinormalisasi. **Hasil:** VectorState — gambaran lengkap kondisi trade saat itu. |
| **Chatbot** | Sistem membangun gambaran keadaan refund: Nilai transaksi, Mode refund, Ketersediaan chargeback state, Authority surface (customer via web). **Hasil:** VectorState refund. |
| **Banking** | Sistem membangun gambaran kondisi pinjaman: Debt ratio bucket, Amount bucket, Authority context (customer via web), Ketersediaan collateral, Kelengkapan dokumen. **Hasil:** VectorState pinjaman. |

### Stage 6.0 — COVERAGE (V1)

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Sistem bertanya: Apakah semua informasi penting tersedia untuk menilai trade ini? **Hasil:** `COVERED` (cukup data) atau `UNCOVERED` (ada data penting yang hilang). |
| **Chatbot** | Sistem bertanya: Apakah semua informasi penting untuk memproses refund tersedia? Contoh: Apakah status chargeback diketahui? Apakah original payment state diketahui? **Hasil:** `COVERED` atau `UNCOVERED`. |
| **Banking** | Sistem bertanya: Apakah semua informasi penting untuk menilai pinjaman tersedia? Contoh: Apakah collateral valuation diketahui? Apakah income terverifikasi? **Hasil:** `COVERED` atau `UNCOVERED`. |

### Stage 7.0 — RULES (V3 Sector-Specific)

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Kalau `COVERED`: Sistem menerapkan kebijakan risiko — Apakah volatilitas terlalu tinggi? Apakah sinyal menunjukkan risiko berlebih? Apakah strategi melanggar policy? **Hasil:** `PERMIT` / `BLOCK` / `REQUIRE_OVERRIDE`. Jika `UNCOVERED` → Tidak dievaluasi. |
| **Chatbot** | Kalau `COVERED`: Sistem menerapkan kebijakan refund — Apakah masih dalam refund window? Apakah nominal di bawah threshold? Apakah metode pembayaran mendukung refund? **Hasil:** `PERMIT` / `BLOCK` / `REQUIRE_OVERRIDE`. Jika `UNCOVERED` → Tidak dievaluasi. |
| **Banking** | Kalau `COVERED`: Sistem menerapkan kebijakan kredit — Apakah debt ratio terlalu tinggi? Apakah jumlah pinjaman melewati threshold? Apakah risiko sesuai profil bank? **Hasil:** `PERMIT` / `BLOCK` / `REQUIRE_OVERRIDE`. Jika `UNCOVERED` → Tidak dievaluasi. |

### Stage 8.0 — SEAL (Evidence Pack)

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Sistem menyegel keputusan: Status epistemic (`COVERED`/`UNCOVERED`), Status authority (`PERMIT`/`BLOCK`/`REQUIRE_OVERRIDE`). Semua dikemas sebagai bukti minimal. **Hasil:** Evidence Pack resmi. |
| **Chatbot** | Sistem menyegel hasil: EP (`COVERED`/`UNCOVERED`), AAT (`PERMIT`/`BLOCK`/`REQUIRE_OVERRIDE`/`NOT_EVALUATED`). **Hasil:** EvidencePack resmi. |
| **Banking** | Sistem menyegel: Status epistemic (`COVERED`/`UNCOVERED`), Status authority (`PERMIT`/`BLOCK`/`REQUIRE_OVERRIDE`). **Hasil:** EvidencePack resmi. |

### Stage 9.0 — FINGERPRINT (Merkle Root / Hash)

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Sistem membuat fingerprint kriptografis atas seluruh proses. **Hasil:** Hash unik untuk kasus tersebut. Fingerprint sebagai anchor audit. |
| **Chatbot** | Membuat fingerprint unik dari seluruh proses. **Hasil:** Hash kriptografis. |
| **Banking** | Membuat fingerprint unik dari seluruh proses. **Hasil:** Hash kriptografis. |

### Stage 10.0 — REPLAYABLE (Audit-ready)

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Kasus sekarang bisa diputar ulang dengan aturan yang sama. Kalau diaudit, hasilnya harus identik. **Hasil:** Keputusan siap audit. |
| **Chatbot** | Kasus refund bisa diaudit ulang kapan saja dengan hasil identik. **Hasil:** Keputusan refund siap audit. |
| **Banking** | Kasus pinjaman bisa diaudit ulang kapan saja. **Hasil:** Loan decision audit-ready. |

### Stage 11.0 — SERVING ADAPTER

| Domain | Deskripsi |
|--------|-----------|
| **All** | Serving layer menerima Evidence Pack. Dia **tidak mengubah keputusan**. Hanya mempersiapkan data untuk explanation dan analog reasoning. **Hasil:** Permintaan analisis lanjutan berbasis keputusan final. Stage berikutnya consume: Evidence Pack + VectorState. |

### Stage 12.0 — KNOWLEDGE SNAPSHOT PIN

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Serving layer memilih snapshot pengetahuan yang tepat: Policy version, Historical dataset, Ontology version. Semua snapshot-based dan version-pinned. **Hasil:** Knowledge set. |
| **Chatbot** | Sistem memilih snapshot pengetahuan: Refund policy version, Historical refund dataset, Ontology snapshot. **Hasil:** Knowledge set. |
| **Banking** | Sistem memilih snapshot pengetahuan: Underwriting guideline version, Historical loan dataset, Ontology kredit. **Hasil:** Knowledge set. |

### Stage 13.0 — HPVD RETRIEVAL

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | HPVD mencari kasus historis yang secara struktur mirip: Geometri market, Pola volatilitas, Action class, Authority context. **Hasil:** Beberapa analog family dengan similarity terkalibrasi. |
| **Chatbot** | HPVD mencari analog historis yang secara struktur mirip: Refund incomplete safety, Refund outside window, Refund abuse pattern, High authority escalation. **Hasil:** Analog family + similarity + confidence interval. |
| **Banking** | HPVD mencari analog historis: Loan dengan debt ratio tinggi, Loan borderline risk, Loan incomplete documentation, High-authority escalation family. **Hasil:** Analog family + similarity + confidence interval. |

### Stage 14.0 — PHASE CONSISTENCY FILTER

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | HPVD memastikan kasus dibandingkan pada fase yang sama. Contoh: Trade execution tidak boleh dibandingkan dengan refund atau payout. **Hasil:** Analog yang benar-benar sejalan secara struktur. |
| **Chatbot** | Refund submission tidak boleh dibandingkan dengan payment capture atau settlement. **Hasil:** Analog yang benar-benar sejalan secara struktur. |
| **Banking** | Loan submission tidak dibandingkan dengan loan repayment. **Hasil:** Analog yang benar-benar sejalan secara struktur. |

### Stage 15.0 — ANALOG FAMILY FORMATION

| Domain | Deskripsi |
|--------|-----------|
| **Finance** | Sistem mengelompokkan kasus ini ke dalam keluarga analog. Contoh: "High volatility escalation family". **Hasil:** Family ID + similarity + confidence interval. |
| **Chatbot** | Contoh: "Refund incomplete verification family". **Hasil:** Family ID + calibrated similarity. |
| **Banking** | Contoh: "High debt-ratio escalation family". **Hasil:** Family ID + calibrated similarity. |

### Stage 16.0 — PMR HYPOTHESIS GRAPH BUILD

| Domain | Deskripsi |
|--------|-----------|
| **All** | PMR membangun grafik hipotesis: Bukti yang mendukung analog, Bukti yang bertentangan, Pola outcome historis. **Hasil:** Structured reasoning graph dengan tingkat confidence. |

### Stage 17.0 — STRUCTURED REASONING OUTPUT

| Domain | Deskripsi |
|--------|-----------|
| **All** | Sistem menggabungkan: Deterministic truth dari Core, Analog similarity dari HPVD, Historical outcome dari PMR. **Hasil:** Objek reasoning lengkap dengan uncertainty. |

### Stage 18.0 — LLM EXPLANATION RENDER

| Domain | Contoh Output |
|--------|---------------|
| **Finance** | "Trade ini berada pada kondisi volatilitas tinggi dan memiliki 0.71 calibrated similarity terhadap analog family 'High Volatility Escalation' (snapshot vX). Dalam keluarga tersebut, 64% historis kasus berujung pada manual override." |
| **Chatbot** | "Sistem belum dapat memproses refund karena informasi belum lengkap. Kasus ini memiliki 0.74 similarity terhadap analog family 'Incomplete Verification' (snapshot vX). Dalam family tersebut, 61% historis kasus memerlukan verifikasi." |
| **Banking** | "Aplikasi pinjaman ini memiliki 0.69 similarity terhadap analog family 'High Debt-Ratio Escalation' (snapshot vX). Dalam family tersebut, 58% historis kasus memerlukan manual review. Confidence interval: [0.52–0.75]." |

---

## 2. Tech Flow

Implementasi teknis dari setiap stage, termasuk input/output dan J-files yang dihasilkan.

### Stage 1.0 — UPLOAD / INPUT

<details>
<summary><strong>Finance</strong></summary>

**Proses:** Finance Adapter — Validate required fields, Validate asset exists, Resolve tenant config, Attach correlation_id, Create execution_request object

**Input:**
```json
{
  "tenant_id": "FINANCE_DESK",
  "asset_id": "AAPL",
  "as_of_date": "1993-05-07",
  "trader_id": "TRADER_005",
  "strategy_id": "MEAN_REVERSION_V1"
}
```

**Output:** `execution_request`
```json
{
  "request_id": "req_001",
  "tenant_id": "...",
  "input_payload": {},
  "received_at": "UTC_TIMESTAMP"
}
```
</details>

<details>
<summary><strong>Chatbot</strong></summary>

**Proses:** Chatbot Layer
- Step 1: Intent Detection — Detect action: `REFUND_REQUEST`, Extract order_id candidate
- Step 2: Slot Filling — Bot may ask: refund_mode? payment method? order_channel? merchant_region?
- Step 3: Conversation State Object as the output

**Input:**
```json
{
  "conversation_id": "conv_991",
  "user_id": "cust_447",
  "channel": "WEB_CHAT",
  "message": "Saya mau refund order #ORD-7782. Barangnya rusak.",
  "timestamp": "UTC"
}
```

**Output:**
```json
{
  "conversation_state": {
    "action_kind": "REFUND",
    "order_id": "ORD-7782",
    "refund_mode": "FULL",
    "payment_method": "CARD",
    "merchant_region": "EU"
  }
}
```
</details>

<details>
<summary><strong>Banking</strong></summary>

**Proses:** Banking Adapter — OCR for uploaded file, Validate schema, Attach correlation_id, Freeze request timestamp, Build loan_request_context

**Input:**
```json
{
  "tenant_id": "BANK_A",
  "applicant_id": "CUST_8891",
  "loan_type": "PERSONAL_LOAN",
  "requested_amount_minor": 500000000,
  "currency": "IDR",
  "term_months": 36,
  "declared_income_minor": 150000000,
  "existing_debt_minor": 60000000,
  "collateral_provided": false
}
```

**Output:** `loan_request_context`
```json
{
  "request_id": "loan_req_001",
  "tenant_id": "BANK_A",
  "input_payload": {},
  "received_at": "UTC"
}
```
</details>

### Stage 2.0 — IMMUTABLE SNAPSHOT

<details>
<summary><strong>Finance</strong></summary>

**Proses:** Finance Adapter — Fetch market data snapshot (read-only DB), Freeze dataset at as_of_date, Normalize (Decimal → string fixed-point, Boolean explicit), Generate UNKNOWN array for missing fields, Sort all keys lexicographically, Build snapshot_payload

**Input:** `execution_request`

**Output:** `snapshot_payload`
```json
{
  "observed": {
    "p0_rv_short": "0.00411076672674",
    "p0_rv_long": "0.0060904626252",
    "p1_entropy_density": "0.71308988303"
  },
  "availability": {
    "has_volatility_structure_known": true
  },
  "unknown_fields": []
}
```
</details>

<details>
<summary><strong>Chatbot</strong></summary>

**Proses:** Chatbot Adapter — Normalize enums, Convert currency → minor units, Explicit UNKNOWN enumeration, Freeze conversation facts

**Input:** `conversation_state`

**Output:** `snapshot_payload`
```json
{
  "observed": {
    "action_kind": "REFUND",
    "order_id": "ORD-7782",
    "amount_minor": 12900,
    "currency": "EUR",
    "refund_mode": "FULL",
    "order_channel": "WEB",
    "payment_method": "CARD",
    "merchant_region": "EU",
    "customer_present": false
  }
}
```
</details>

<details>
<summary><strong>Banking</strong></summary>

**Proses:** Banking Adapter — Normalize: All monetary fields → fixed-point string, Compute DTI ratio: `dti = existing_debt / declared_income`, Enumerate UNKNOWN fields, Freeze application facts

**Input:** `loan_request_context`

**Output:** `snapshot_payload`
```json
{
  "observed": {
    "loan_type": "PERSONAL_LOAN",
    "requested_amount_minor": "500000000",
    "term_months": 36,
    "declared_income_minor": "150000000",
    "existing_debt_minor": "60000000",
    "dti_ratio": "0.40",
    "collateral_provided": false
  },
  "availability": {}
}
```
</details>

### Stage 3.0 — CASE METADATA CAPTURE

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **Finance** | Core Entry Service — Generate deterministic commit_id, Validate reentrancy_guard, Build J01. *Pack Init juga generate J02 (diagnostic).* | `snapshot_payload` + `execution_request` | **J01** |
| **Chatbot** | Core Entry — Generate deterministic commit_id: `sha256(tenant + order_id + canonical(snapshot))`, Enforce `SINGLE_CAPTURE_ENFORCED`, Attach boundary. *Pack Init juga generate J02.* | `snapshot_payload` + tenant context | **J01** |
| **Banking** | Core Entry — Deterministic commit_id: `sha256(tenant + applicant_id + canonical(snapshot))`, Enforce `SINGLE_CAPTURE_ENFORCED`, Attach boundary: `LOAN_SUBMISSION_COMMIT_T0`. *Pack Init juga generate J02.* | `snapshot_payload` + tenant context | **J01** |

### Stage 4.0 — STRUCTURED CASE JSON

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Canonical sort keys, Remove transient fields, Serialize, Compute `canonical_bytes_hash`. *Capture Receipt juga generate J04 (acknowledgement).* | **J01** | **J03** |

### Stage 5.0 — DETERMINISTIC PROJECTION (Merge Metadata + Doc)

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | **Step A — StructuredContext (J05):** Extract authority_identity, intent, domain_observed, availability normalized. **Step B — VectorState (J06):** Build metadata, domain_state.metrics, domain_state.flags, unknown_bitmap, structural_state derived. *Optional: Trajectory Tracking → J07* | **J03** | **J05**, **J06**, *(J07)* |

### Stage 6.0 — COVERAGE (V1)

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **Finance** | Load `required_set.finance.v1`, Check required fields, Propagate unknown_bitmap, Determine EP | **J06** | **J08** |
| **Chatbot** | Check required fields for refund: order_id, amount, payment_method, chargeback state known? If unknown → EP = `UNCOVERED` | **J06** | **J08** |
| **Banking** | Required fields: requested_amount, term_months, declared_income, credit_score. If unknown → EP = `UNCOVERED` | **J06** | **J08** |

### Stage 7.0 — RULES (V3 Sector-Specific)

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **Finance** | Load `ruleset_finance.vX`, Validate ruleset_hash, Evaluate thresholds, Determine AAT | **J06** + **J08** | **J09** |
| **Chatbot** | If EP = `UNCOVERED`: AAT = `NOT_EVALUATED`. Else evaluate PSP rule. | **J06** + **J08** | **J09** |
| **Banking** | If EP = `UNCOVERED`: AAT = `NOT_EVALUATED`. Else: Evaluate DTI threshold, Evaluate credit score, Possibly `REQUIRE_OVERRIDE` | **J06** + **J08** | **J09** |

### Stage 8.0 — SEAL (Evidence Pack)

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Build Merkle tree, Compute hash_root, Generate pack_id | **J03** + **J06** + **J08** + **J09** | **J12** |

### Stage 9.0 — FINGERPRINT + Stage 10.0 — REPLAYABLE

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Hash root verification, Replayable record. Compare: canonical_bytes_hash, vector_hash, EP, AAT, Merkle root | **J01–J12** | **J20** |

### Stage 11.0 — SERVING ADAPTER

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Step 1: Replay Validation Gate — If replay ≠ PASS → ABORT. Step 2: Build ServingContext. Step 3: Generate opaque_pack_ref | **J12** + **J20** | **J13** |

### Stage 12.0 — KNOWLEDGE SNAPSHOT PIN

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Resolve pinset_snapshot_id, Load: dataset_snapshot_id, ontology_version, calibration_model_version. Validate snapshot immutable, Attach to retrieval context | **J13** | *(internal object, no J-file)* |

### Stage 13.0 — HPVD RETRIEVAL

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Structural-first analog retrieval. Extract action_class, vector geometry reference. Retrieve candidate analogs. Compute calibrated_similarity. Compute confidence interval. Evaluate abstention. | **J13** + Knowledge snapshot | **J14** |

### Stage 14.0 — PHASE CONSISTENCY FILTER

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Compare phase_label vs current event lifecycle, Reject mismatch, Preserve reason | **J14** | **J15** |

### Stage 15.0 — ANALOG FAMILY FORMATION

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Assign cluster_id / family_id, Compute membership_probability, Retrieve historical distribution | **J15** | **J16** |

### Stage 16.0 — PMR HYPOTHESIS GRAPH BUILD

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **Finance** | Build hypothesis nodes (VOL_RATIO_THRESHOLD: support, LOW_ENTROPY_STABILITY: contradiction), Compute weight, Compute overall_confidence | **J16** | **J17** |
| **Chatbot** | Build hypothesis nodes (CHARGEBACK_UNKNOWN: support, PSP_CAPABILITY_KNOWN: contradiction), Compute weight, Compute overall_confidence | **J16** | **J17** |
| **Banking** | Build hypothesis nodes (DTI_RATIO_HIGH: support, STRONG_COLLATERAL: contradiction), Compute weight, Compute overall_confidence | **J16** | **J17** |

### Stage 17.0 — STRUCTURED REASONING OUTPUT

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Extract EP + AAT, Attach analog distribution, Attach uncertainty, **DO NOT override AAT**, Add provenance snapshot | **J12** + **J16** + **J17** | **J18** |

### Stage 18.0 — LLM EXPLANATION RENDER

| Domain | Proses | Input | Output |
|--------|--------|-------|--------|
| **All** | Build prompt from J18, Inject disclaimer, No policy suggestion, No override, Must cite pack_ref | **J18** | **J19** |

---

## 3. J-Files Reference

Setiap J-file merepresentasikan artefak terstruktur pada satu tahap pipeline.

### J01 — CommitBoundaryEvent

<details>
<summary>Finance</summary>

```json
{
  "kind": "J01.CommitBoundaryEvent",
  "schema_id": "manithy.commit_boundary_event.v2",
  "commit_id": "sha256_trade_commit",
  "commit_seq": 1,
  "meta": {
    "tenant_id": "FINANCE_DESK",
    "action_class": "TRADE_EXECUTION",
    "commit_point_id": "TRADE_EXECUTION_COMMIT_T_MINUS_1"
  },
  "boundary": {
    "boundary_kind": "...",
    "boundary_seq": 1,
    "same_thread": true,
    "reentrancy_guard": "SINGLE_CAPTURE_ENFORCED"
  }
}
```
</details>

<details>
<summary>Chatbot</summary>

```json
{
  "kind": "J01.CommitBoundaryEvent",
  "schema_id": "manithy.commit_boundary_event.v2",
  "commit_id": "sha256_refund_commit",
  "commit_seq": 1,
  "meta": {
    "tenant_id": "MERCHANT_EU",
    "action_class": "CHATBOT_EXECUTION",
    "commit_point_id": "REFUND_COMMIT_T_MINUS_1"
  },
  "boundary": {
    "boundary_kind": "...",
    "boundary_seq": 1,
    "same_thread": true,
    "reentrancy_guard": "SINGLE_CAPTURE_ENFORCED"
  }
}
```
</details>

<details>
<summary>Banking</summary>

```json
{
  "kind": "J01.CommitBoundaryEvent",
  "schema_id": "manithy.commit_boundary_event.v2",
  "commit_id": "sha256_loan_commit",
  "commit_seq": 1,
  "meta": {
    "tenant_id": "BANKING_CORE",
    "action_class": "LOAN_EXECUTION",
    "commit_point_id": "LOAN_SUBMISSION_COMMIT_T_MINUS_1"
  },
  "boundary": {
    "boundary_kind": "...",
    "boundary_seq": 1,
    "same_thread": true,
    "reentrancy_guard": "SINGLE_CAPTURE_ENFORCED"
  }
}
```
</details>

### J02 — PackInit

<details>
<summary>Finance</summary>

```json
{
  "kind": "J02.PackInit",
  "commit_id": "sha256_trade_commit",
  "ingress_timestamp_pinned": "1993-05-07",
  "adapter_status": "CAPTURED",
  "kill_switch_state": "OFF",
  "kernel_build_id": "core@2.4.1",
  "vector_schema_version": "v2",
  "ruleset_version": "r1",
  "policy_bundle_version": "p1"
}
```
</details>

<details>
<summary>Chatbot</summary>

```json
{
  "kind": "J02.PackInit",
  "commit_id": "sha256_refund_commit",
  "ingress_timestamp_pinned": "2026-02-25",
  "adapter_status": "CAPTURED",
  "kill_switch_state": "OFF",
  "kernel_build_id": "core@2.4.1",
  "vector_schema_version": "v2",
  "ruleset_version": "r1",
  "policy_bundle_version": "p1"
}
```
</details>

<details>
<summary>Banking</summary>

```json
{
  "kind": "J02.PackInit",
  "commit_id": "sha256_loan_commit",
  "ingress_timestamp_pinned": "2026-02-25",
  "adapter_status": "CAPTURED",
  "kill_switch_state": "OFF",
  "kernel_build_id": "core@2.4.1",
  "vector_schema_version": "v2",
  "ruleset_version": "r1",
  "policy_bundle_version": "p1"
}
```
</details>

### J03 — CCR (Canonical Case Record)

Canonical representation of the case with `canonical_bytes_hash` for integrity.

### J04 — CaptureReceipt

```json
{
  "kind": "J04.CaptureReceipt",
  "commit_id": "sha256_<domain>_commit",
  "ack_status": "ACCEPTED",
  "duplicate_detected": false,
  "idempotency_anchor": "sha256_<domain>_commit"
}
```

### J05 — StructuredContext

Extracted authority_identity, intent, domain_observed, and availability normalized from J03.

### J06 — VectorState

Full domain-specific vector representation (see [Format VectorState](#4-format-vectorstate) section).

### J07 — Trajectory

```json
{
  "kind": "J07.Trajectory",
  "schema_id": "manithy.trajectory.v2",
  "commit_id": "sha256_<domain>_commit",
  "steps": [
    { "seq": 1, "stage": "CCR_SEALED" },
    { "seq": 2, "stage": "CONTEXT_NORMALIZED" },
    { "seq": 3, "stage": "VECTORSTATE_BUILT" },
    { "seq": 4, "stage": "STRUCTURAL_STATE_DERIVED" },
    { "seq": 5, "stage": "V1_EVALUATED" }
  ],
  "flags": {
    "unknown_propagation": true,
    "bounded_window_ok": true,
    "deterministic_path": true
  }
}
```

### J08 — L1_Admissibility

```json
// Case: COVERED
{
  "kind": "J08.L1_Admissibility",
  "schema_id": "manithy.v1.epistemic_attestation.v2",
  "commit_id": "sha256_<domain>_commit",
  "vector_hash": "sha256_vector_<domain>",
  "ep": "COVERED",
  "closed_reason": "ALL_REQUIRED_COORDS_KNOWN",
  "required_set_id": "reqset.<domain>.vX"
}

// Case: UNCOVERED
{
  "ep": "UNCOVERED",
  "closed_reason": "REQUIRED_FIELD_UNKNOWN"
}
```

### J09 — EligibilityFeatureVector (EFV)

<details>
<summary>Finance</summary>

```json
{
  "kind": "J09.EligibilityFeatureVector",
  "schema_id": "manithy.efv.finance.v5",
  "commit_id": "sha256_trade_commit",
  "features": {
    "F0_action_class": "TRADE_EXECUTION",
    "F1_volatility_bucket": "LOW",
    "F2_entropy_bucket": "MEDIUM",
    "F3_channel": "SYSTEM",
    "F4_risk_flag": false
  }
}
```
</details>

<details>
<summary>Chatbot</summary>

```json
{
  "kind": "J09.EligibilityFeatureVector",
  "schema_id": "manithy.efv.chatbot.v3",
  "commit_id": "sha256_refund_commit",
  "features": {
    "F0_action_class": "CHATBOT_EXECUTION",
    "F1_amount_bucket": "B_100_199_EUR",
    "F2_channel": "WEB",
    "F3_region": "EU",
    "F4_chargeback_known": true
  }
}
```
</details>

<details>
<summary>Banking</summary>

```json
{
  "kind": "J09.EligibilityFeatureVector",
  "schema_id": "manithy.efv.banking.v2",
  "commit_id": "sha256_loan_commit",
  "features": {
    "F0_action_class": "LOAN_EXECUTION",
    "F1_amount_bucket": "B_500K_1M_USD",
    "F2_debt_ratio_bucket": "MEDIUM",
    "F3_income_verified": true,
    "F4_collateral_known": true
  }
}
```
</details>

### J10 — AuthorityAttestationToken (AAT)

Possible values per EP status:

| EP Status | Possible AAT |
|-----------|-------------|
| `COVERED` | `PERMIT`, `BLOCK`, `REQUIRE_OVERRIDE` |
| `UNCOVERED` | `NOT_EVALUATED` |

```json
{
  "kind": "J10.AuthorityAttestationToken",
  "schema_id": "manithy.v3.aat.v2",
  "commit_id": "sha256_<domain>_commit",
  "aat": "PERMIT | BLOCK | REQUIRE_OVERRIDE | NOT_EVALUATED",
  "ruleset_id": "ruleset.<domain>.vX",
  "ruleset_hash": "sha256_ruleset_<domain>_vX"
}
```

### J11 — EvidencePack

```json
{
  "kind": "J11.EvidencePack",
  "schema_id": "manithy.evidence_pack.v5",
  "pack_id": "<TENANT>:<SUBJECT>:<SEQ>",
  "commit_id": "sha256_<domain>_commit",
  "hash_root": "sha256_merkle_root_<domain>",
  "attestations": {
    "ep": "COVERED | UNCOVERED",
    "aat": "PERMIT | BLOCK | REQUIRE_OVERRIDE | NOT_EVALUATED"
  },
  "provenance": {
    "vector_schema_id": "manithy.vector_state.<domain>.v2",
    "ruleset_id": "ruleset.<domain>.vX",
    "ruleset_hash": "sha256_ruleset_<domain>_vX",
    "ccr_schema_id": "manithy.ccr.<domain>.v2",
    "vector_map_id": "vmap.<domain>.vX",
    "core_build_id": "core@2.4.1"
  }
}
```

### J12 — V2_FactsEnvelope

```json
{
  "kind": "J12.V2_FactsEnvelope",
  "schema_id": "manithy.v2.facts_envelope.v2",
  "pack_ref": {
    "pack_id": "<TENANT>:<SUBJECT>:<SEQ>",
    "hash_root": "sha256_merkle_root_<domain>",
    "pinset_snapshot_id": "PINSET_2026W06"
  },
  "facts": {
    "event_class": "TRADE_EXECUTION | PAYMENT_REFUND | LOAN_SUBMISSION",
    "ep": "COVERED | UNCOVERED",
    "aat": "PERMIT | BLOCK | REQUIRE_OVERRIDE | NOT_EVALUATED"
  },
  "rotation": {
    "pseudonym_epoch": "EPOCH_2026W06",
    "rotating_join_key": "rot_xxxxx"
  }
}
```

### J20 — ReplayReport

```json
{
  "schema_id": "manithy.replay.report.v2",
  "pack_id": "pack_01<DOMAIN>X9K2KQ9G7VZ0N5M",
  "replay": "PASS",
  "matched": {
    "pinset_id": "pinset_<domain>.vX",
    "ruleset_id": "ruleset_<domain>.vX",
    "verifier_hash": "sha256:xxxx...xxxx",
    "core_build_id": "core@xxxxxxx"
  },
  "notes": "FORBIDDEN"
}
```

### J13 — PostCoreQuery

```json
{
  "schema_id": "manithy.post_core_query.v2",
  "binding": "NON_BINDING",
  "query_id": "Q_<ACTION>_SUPPORT",
  "query_version": 1,
  "opaque_pack_ref": {
    "pack_id": "pack_01<DOMAIN>X9K2KQ9G7VZ0N5M",
    "pinset_snapshot_id": "pinset_<domain>.vX#snap_N"
  },
  "scope": {
    "allowed_topics": ["..."],
    "allowed_corpora": ["..."],
    "allowed_doc_types": ["PDF", "POLICY_TEXT", "MARKDOWN"],
    "temporal_scope": "LAST_365_DAYS",
    "max_results": 10,
    "citation_policy": "CITE_REQUIRED"
  }
}
```

### J14 — HPVD_RetrievalRaw

```json
{
  "schema_id": "manithy.hpvd.retrieval_raw.v2",
  "binding": "NON_BINDING",
  "query_id": "Q_<ACTION>_SUPPORT",
  "candidates": [
    {
      "doc_id": "...",
      "chunk_id": "cXX",
      "calibrated_similarity": 0.84,
      "confidence_interval": [0.78, 0.89],
      "phase_label": "EXECUTION_PHASE",
      "abstention_flag": false
    }
  ],
  "lineage": {
    "knowledge_snapshot": "ksnap_<domain>_2026_02_01",
    "retrieval_config_id": "hpvd_cfg_<domain>_vX"
  }
}
```

### J15 — PhaseFilteredSet

```json
{
  "schema_id": "manithy.hpvd.phase_filtered.v1",
  "binding": "NON_BINDING",
  "accepted": [
    {
      "doc_id": "...",
      "chunk_id": "cXX",
      "calibrated_similarity": 0.84
    }
  ],
  "rejected": [
    {
      "doc_id": "...",
      "reason": "PHASE_MISMATCH"
    }
  ]
}
```

### J16 — AnalogFamilyAssignment

```json
{
  "schema_id": "manithy.hpvd.analog_family.v1",
  "binding": "NON_BINDING",
  "family_id": "AF_<PATTERN>_VX",
  "membership_probability": 0.68,
  "confidence_interval": [0.60, 0.75],
  "cluster_snapshot_id": "cluster_<domain>_vX",
  "family_characteristics": {
    "dominant_pattern": "...",
    "historical_outcome_distribution": {
      "PERMIT": 0.21,
      "REQUIRE_OVERRIDE": 0.63,
      "BLOCK": 0.16
    }
  }
}
```

### J17 — PMR_HypothesisGraph

```json
{
  "schema_id": "manithy.pmr.hypothesis_graph.v1",
  "binding": "NON_BINDING",
  "family_id": "AF_<PATTERN>_VX",
  "nodes": [
    { "node_id": "...", "type": "SUPPORT", "weight": 0.42 },
    { "node_id": "...", "type": "CONTRADICTION", "weight": 0.18 }
  ],
  "overall_confidence": 0.70,
  "uncertainty": {
    "confidence_interval": [0.62, 0.78],
    "abstention_flag": false
  }
}
```

### J18 — StructuredReasoningOutput

```json
{
  "schema_id": "manithy.serving.reasoning_output.v2",
  "binding": "NON_BINDING",
  "pack_ref": { "pack_id": "pack_01<DOMAIN>X9K2KQ9G7VZ0N5M" },
  "deterministic_truth": {
    "ep": "COVERED | UNCOVERED",
    "aat": "PERMIT | BLOCK | REQUIRE_OVERRIDE | NOT_EVALUATED"
  },
  "analog_grounding": {
    "family_id": "AF_<PATTERN>_VX",
    "membership_probability": 0.68,
    "confidence_interval": [0.60, 0.75]
  },
  "historical_distribution": {
    "PERMIT": 0.21,
    "REQUIRE_OVERRIDE": 0.63,
    "BLOCK": 0.16
  },
  "uncertainty": {
    "pmr_confidence": 0.70,
    "confidence_interval": [0.62, 0.78],
    "abstention_flag": false
  },
  "provenance": {
    "knowledge_snapshot": "ksnap_<domain>_2026_02_01"
  }
}
```

### J19 — Renderer_Output

```json
{
  "schema_id": "manithy.renderer.output.v2",
  "label": "NON_BINDING",
  "disclaimer": "Support-only narrative. Not evidence. Not a permission/denial.",
  "text": "..."
}
```

---

## 4. Format VectorState

VectorState (J06) adalah representasi vektor lengkap dari satu kasus. Strukturnya terdiri dari 6 bagian:

### 4.1 metadata (kernel identity)

```yaml
meta:
  vector_schema_version: "v2"
  ruleset_version: "r1"
  policy_bundle_version: "p1"
  commit_id: "..."
  commit_point_id: "..."
  tenant_id: "..."
  action_class: "TRADE_EXECUTION | CHATBOT_EXECUTION | LOAN_EXECUTION"
```

### 4.2 boundary

```yaml
boundary:
  boundary_kind: "..."
  boundary_seq: 1
  same_thread: true
  reentrancy_guard: "SINGLE_CAPTURE_ENFORCED"
```

### 4.3 authority identity

```yaml
authority_identity:
  channel: ENUM
  actor_role: ENUM
  agent_identity: ENUM
  auth_level: ENUM
  session_trust_level: ENUM
  actor_hash: STRING
```

### 4.4 intent

```yaml
intent:
  action_kind: STRING
  subject_key: STRING
  subject_hash: STRING
  irreversible: true
```

| Domain | action_kind | subject_key |
|--------|-------------|-------------|
| Finance | `TRADE_EXECUTION` | `TRADER_005` |
| Chatbot | `REFUND` | `ORDER_67250` |
| Banking | `LOAN_SUBMISSION` | `APPLICATION_8891` |

### 4.5 domain state

Bagian ini **berbeda per domain**:

<details>
<summary><strong>Finance</strong></summary>

```yaml
domain_state:
  p0:
    metrics:
      rv_short: 0.0041107
      rv_long: 0.0060904
      vol_ratio: 0.67495
      vol_of_vol: 0.00072806
      amihud_illiquidity: 3.10e-08
      state_velocity: 0
      regime_persistence: 0
    flags:
      proc_fail: false
  p1:
    metrics:
      K: 3.50e-05
      LCV: 6.25e-06
      LTV: 6.82e-06
      entropy_density: 0.71308
    flags:
      K_EXCESS: false
      LCV_SPIKE: false
      LTV_SHOCK: false
      IDD: false
      POPPER_FALSIFIABLE: true
  availability:
    liquidity_proxy_known: true
    volatility_structure_known: true
    curvature_signal_known: true
```
</details>

<details>
<summary><strong>Chatbot</strong></summary>

```yaml
domain_state:
  p0:
    metrics:
      amount_minor: 12900
    flags:
      customer_present: false
      operator_initiated: false
  p1:
    metrics: {}
    flags: {}
  availability:
    original_payment_state_known: true
    psp_refund_capability_known: true
    chargeback_state_known: false
```
</details>

<details>
<summary><strong>Banking</strong></summary>

```yaml
domain_state:
  p0:
    metrics:
      requested_amount_minor: 5000000
      income_minor: 200000
      debt_ratio: 0.35
    flags:
      collateral_present: true
  p1:
    metrics:
      credit_signal_entropy: 0.61
    flags:
      income_volatility_spike: false
  availability:
    income_verified: true
    bureau_data_known: true
    collateral_valuation_known: false
```
</details>

### 4.6 structural state

```yaml
structural_state:
  trajectory_length_bucket: ENUM
  commit_density_bucket: ENUM
  authority_stability_flag: BOOLEAN
  repeated_missing_flag: BOOLEAN
  entropy_bucket: ENUM
```

---

## 5. Task Assignment

| Task | Priority | Responsibility | What Already Done | Adjustment Needed | Expected Time |
|------|:--------:|----------------|-------------------|-------------------|:-------------:|
| Build banking ADAPTER | 1 | Ibrahim | OCR | Build all dependencies | 1 week |
| Revise CORE layer | 1 | Kaira | Core schema | Refine JSON contract with new vectorstate form | 1 week |
| Connect adapter banking to CORE | 2 | Kaira | - | Working with banking adapter result | 1 week |
| Manage HPVD | 2 | Junpito | Raw module | Change mostly HPVD part in the input/output definition based on consumed data from KL | 2 weeks |
| Manage PMR DB | 2 | Fitria | Raw module | Change mostly PMR DB part in the input/output, form response | 2 weeks |
| Build KL | - | Arfiano Dzaky | - | Consume data from KL | - |

---

## Pipeline Visual Summary

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. INPUT    │────▶│  2. SNAPSHOT  │────▶│  3. METADATA │────▶│  4. CCR      │
│  (Adapter)   │     │  (Immutable)  │     │  (J01)       │     │  (J03)       │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
       ┌───────────────────────────────────────────────────────────────┘
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  5. PROJECT  │────▶│  6. COVERAGE │────▶│  7. RULES    │────▶│  8. SEAL     │
│  (J05+J06)   │     │  V1 (J08)    │     │  V3 (J09)    │     │  (J11/J12)   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
       ┌───────────────────────────────────────────────────────────────┘
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 9. FINGERPRINT────▶│10. REPLAYABLE│────▶│11. SERVING   │────▶│12. KNOWLEDGE │
│  (J20)       │     │  (Audit)     │     │  ADAPTER(J13)│     │  SNAPSHOT PIN│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
       ┌───────────────────────────────────────────────────────────────┘
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│13. HPVD      │────▶│14. PHASE     │────▶│15. ANALOG    │────▶│16. PMR       │
│ RETRIEVAL(J14)     │ FILTER (J15) │     │ FAMILY (J16) │     │ GRAPH (J17)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
       ┌───────────────────────────────────────────────────────────────┘
       ▼
┌──────────────┐     ┌──────────────┐
│17. REASONING │────▶│18. LLM       │
│ OUTPUT (J18) │     │ RENDER (J19) │
└──────────────┘     └──────────────┘
```

---

> **Catatan:** J-files J02 (PackInit), J04 (CaptureReceipt), dan J07 (Trajectory) bersifat **diagnostik/opsional** — dihasilkan di dalam stage tertentu untuk keperluan audit dan debugging, bukan sebagai output utama pipeline.
