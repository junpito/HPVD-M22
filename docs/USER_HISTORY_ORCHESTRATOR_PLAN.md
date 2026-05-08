# Perencanaan: User History Retrieval & Orchestrator untuk HPVD

> **Status:** Draft - menunggu konfirmasi
> **Dibuat:** April 2026  
> **Konteks:** Ide tambahan bukan urgent, dikerjakan setelah HPVD production stabil

---

## 1. Latar Belakang

Saat ini HPVD berjalan sebagai **stateless query service** - setiap request berdiri sendiri tanpa "ingatan" tentang submission sebelumnya. Pertanyaan yang muncul:

> *Jika seorang user sudah pernah submit case sebelumnya (datanya tersimpan di KL), apakah HPVD bisa retrieve data terkait user tersebut saat user submit case lagi?*

**Jawaban singkat:** Saat ini belum bisa secara native. HPVD hanya mencocokkan `observed_data` dari Parser dengan corpus knowledge (Policy/Product/RuleMapping) berdasarkan `sector`. Tidak ada mekanisme lookup by user ID.

---

## 2. Pemahaman Arsitektur Saat Ini

### 2.1 Posisi HPVD dalam Pipeline Manithy

```
[NRB]
  Parser -> HPVD -> PMR -> Knowledge Builder
               |
          t-1 boundary
               |
[Core]
  J01 -> ... -> J06(VectorState) -> ... -> J12 -> J13(PostCoreQuery)
                                                       |
                                          Multi-Manifold + Geometry (DGG/AIR)
```

HPVD beroperasi di **NRB (Non-Binding Realm)** - sebelum Core. Multi-manifold untuk chat generation ada **setelah Core**, menggunakan output dari seluruh pipeline termasuk HPVD.

### 2.2 Cara HPVD Retrieve Saat Ini

HPVD menerima input:
```json
{
  "commit_id": "...",
  "sector": "banking",
  "observed": { "loan_amount": 50000000, "income": 10000000 },
  "availability": {}
}
```

Dan mencocokkan field `observed` dengan corpus KL yang dimuat saat startup (stateless). **Tidak ada filter by user ID.**

### 2.3 Apa yang Dimuat dari KL

`KLCorpusLoader` hanya mengenali 4 tipe objek saat startup:

| Key | Object Type |
|---|---|
| `policy_id` | `policy` |
| `product_id` | `product` |
| `mapping_id` | `rule_mapping` |
| `doc_type` | `document_schema` |

Data user history **tidak akan ikut termuat** karena tidak punya key tersebut.

### 2.4 Output HPVD - J16 Families (Manifold)

J16 sudah merupakan output terkelompok per tipe (inilah "manifold" yang dimaksud):

```json
{
  "families": [
    { "family_id": "knowledge_policy",       "..." : "..." },
    { "family_id": "knowledge_product",      "..." : "..." },
    { "family_id": "knowledge_rule_mapping", "..." : "..." }
  ]
}
```

Untuk **multi-manifold chat generation**: konsumsi J16 families dari output HPVD yang sudah ada, atau panggil HPVD lagi dengan `domain: "document"` untuk semantic search context FAQ/guide.

---

## 3. Analisis KL API (Production)

Berdasarkan `https://knowledge-layer-production.up.railway.app/openapi.json`:

### 3.1 Tidak ada field `user_id` di KL

| Field | Lokasi | Bisa difilter? | Catatan |
|---|---|---|---|
| `created_by` | Document, Event | Tidak | Tidak ada filter by created_by di endpoint search |
| `commit_id` | Event (required) | **Ya** - `GET /events/commits/{commit_id}` | Per case, bukan per user |
| `tag` | DocumentMetadata | **Ya** - `GET /documents?tag=...` | Paling fleksibel untuk user ID |
| `tenant_id` | Document, Event | Tidak | Level organisasi, bukan individual user |

### 3.2 Endpoint Penting yang Belum Dipakai di HPVD

```
GET /events/commits/{commit_id}
```

Bisa retrieve semua events untuk satu `commit_id` - berguna jika satu case = satu commit_id.

### 3.3 EventCreate di KL

```json
{
  "event_kind": "string",
  "commit_id": "string",     // REQUIRED
  "payload":   { ... },      // free JSON, bisa isi apapun
  "created_by": "string"     // optional
}
```

---

## 4. Opsi Implementasi

### Opsi A - Simpan user history di KL sebagai dokumen + tag user ID
*Paling clean jika BE menggunakan KL sebagai storage*

**Alur:**
```
BE simpan:       POST /documents  { tag: "user_uid_123", domain: "user_history" }
Orchestrator:    GET /documents?tag=user_uid_123&domain=user_history
Inject ke:       observed = { ...parser_output, ...user_history_fields }
Kirim ke HPVD:   POST /query dengan observed yang diperkaya
```

**Pro:** Tidak perlu modifikasi HPVD, pakai `KLDocumentLoader.load_with_search()` yang sudah ada  
**Kontra:** Bergantung pada BE untuk set format & tag di KL

---

### Opsi B - User history di database BE, fetch via API
*Jika BE menyimpan di database terpisah (user management)*

**Alur:**
```
BE simpan:       Database user management (Postgres, dll)
Orchestrator:    GET /users/{user_id}/history
Inject ke:       observed = { ...parser_output, ...user_history }
Kirim ke HPVD:   POST /query
```

**Pro:** Tidak coupling ke KL, BE bebas pakai storage apapun  
**Kontra:** Butuh API contract baru dengan BE

---

### Opsi C - KL Events dengan commit_id per submission
*Jika BE sudah menggunakan event system KL*

**Alur:**
```
BE simpan:       create_event(event_kind="case_submitted", commit_id=..., payload={...})
Orchestrator:    GET /events/commits/{commit_id}   // per case
                 (untuk multi-case: perlu index commit_ids per user di luar KL)
```

**Pro:** Pakai infrastruktur event chain KL yang sudah ada  
**Kontra:** Tidak ada filter langsung by user - butuh index eksternal untuk mapping user -> list of commit_ids

---

### Opsi D - Graceful degradation (sambil menunggu kepastian BE)
*Paling aman untuk saat ini*

**Alur:**
```python
try:
    history = fetcher.fetch(user_id)
except:
    history = {}  # HPVD tetap jalan tanpa history

observed = { **parser_output, **history }
# POST /query ke HPVD
```

**Pro:** Bisa jalan sekarang tanpa tunggu BE, zero risk  
**Kontra:** Tidak ada user context di retrieval

---

## 5. Desain Orchestrator yang Diusulkan

Apapun opsi yang dipilih, orchestrator dirancang dengan **abstraction layer** sehingga sumber data bisa diswap tanpa mengubah logic utama atau kode HPVD.

```python
# Abstraction
class UserHistoryFetcher(ABC):
    @abstractmethod
    def fetch(self, user_id: str) -> Dict[str, Any]:
        """Return dict siap di-merge ke observed."""
        ...

# Implementasi A: dari KL via tag
class KLTagHistoryFetcher(UserHistoryFetcher):
    def fetch(self, user_id: str) -> Dict[str, Any]:
        chunks = self._loader.load_with_search(tag=user_id, domain="user_history")
        return self._flatten_chunks(chunks)

# Implementasi B: dari external API
class ExternalAPIHistoryFetcher(UserHistoryFetcher):
    def fetch(self, user_id: str) -> Dict[str, Any]:
        resp = self._http.get(f"/users/{user_id}/history")
        return resp.json()

# Implementasi D: fallback kosong (untuk sekarang)
class NoOpHistoryFetcher(UserHistoryFetcher):
    def fetch(self, user_id: str) -> Dict[str, Any]:
        return {}


# Orchestrator utama
class HPVDOrchestrator:
    def __init__(self, history_fetcher: UserHistoryFetcher):
        self._fetcher = history_fetcher  # swap di sini saat opsi berubah

    def query(self, user_id: str, parser_observed: dict) -> dict:
        try:
            history = self._fetcher.fetch(user_id)
        except Exception:
            history = {}

        enriched_observed = {**parser_observed, **history}

        return self._hpvd_client.post("/query", json={
            "commit_id": ...,
            "sector": ...,
            "observed": enriched_observed,
        })
```

**Keuntungan desain ini:**
- Zero perubahan di HPVD (`observed` sudah `Dict[str, Any]`)
- Swap implementasi cukup ganti 1 baris di inisialisasi
- Bisa mulai dengan `NoOpHistoryFetcher` sekarang

---

## 6. Pertanyaan yang Perlu Dikonfirmasi ke BE

Sebelum memilih opsi, konfirmasi ke BE:

| # | Pertanyaan | Kenapa penting |
|---|---|---|
| 1 | Data hasil case/commit disimpan di KL atau DB lain? | Menentukan opsi A vs B |
| 2 | Jika di KL, identifier apa yang dipakai? (`tag`, `created_by`, atau field di `payload`)? | Menentukan cara fetch |
| 3 | Format data per submission seperti apa? (JSON flat, atau nested?) | Menentukan cara flatten ke `observed` |
| 4 | Apakah user ID sudah ada di request yang masuk ke HPVD? | Kalau belum, perlu extend request schema |
---

## 7. Terkait Multi-Manifold Chat Generation

Untuk penggunaan HPVD di sana:

### Opsi 1 - Konsumsi J16 yang sudah ada (tidak perlu call HPVD lagi)
Output J16 dari HPVD sudah terkelompok per "manifold":
- `knowledge_policy`      -> rules & eligibility
- `knowledge_product`     -> product constraints
- `knowledge_rule_mapping` -> field requirements

Chat generation langsung konsumsi J16 families sebagai context.

### Opsi 2 - Call HPVD dengan `domain: "document"` untuk semantic search
Untuk retrieve FAQ/guide/policy text sebagai context chat:

```json
{
  "scope": { "domain": "document" },
  "query_payload": { "text": "pertanyaan user..." },
  "allowed_topics": ["faq", "guide"],
  "allowed_doc_types": ["FAQ", "GUIDE", "POLICY_TEXT"]
}
```

HPVD akan menggunakan `DocumentRetrievalStrategy` (Sentence-Transformer + FAISS) untuk semantic search.

---
 
*Dibuat berdasarkan analisis kodebase HPVD-M22 dan KL production API -  April 2026*
