# Prompt untuk Melanjutkan Refactoring HPVD ke Manithy

Kapanpun Anda siap dan quota sudah tersedia, cukup *copy-paste* teks di bawah ini ke prompt chat AI Anda untuk langsung mengeksekusi rencana refactoring yang sudah kita sepakati.

---

**COPY TEXT DI BAWAH INI:**

Halo! Silakan baca dokumen ini dan langsung mulai eksekusi refactoring berdasarkan *Implementation Plan* dan *Task List* di bawah. Kita sudah sepakat bahwa:
1. HPVD ini akan murni digunakan untuk **Knowledge dan Document retrieval**.
2. Komponen `HPVDEngine` core (termasuk dense/sparse index, trajectory, distance, dll) dan J-File pipeline **boleh dihapus sepenuhnya**.

Tolong jalankan langkah-langkah di bawah secara bertahap dan buat commit jika memungkinkan, atau minimal pastikan `api.py` dan `pytest` berjalan lancar setelah refactoring selesai.

### IMPLEMENTATION PLAN

**Phase 1: Delete Dead Code**
Kita akan menghapus file-file legacy (finance & core engine) yang sudah tidak terpakai:
- `src/hpvd/synthetic_data_generator.py`
- `src/hpvd/engine.py`
- `src/hpvd/trajectory.py`
- `src/hpvd/sparse_index.py`
- `src/hpvd/dense_index.py`
- `src/hpvd/distance.py`
- `src/hpvd/dna_similarity.py`
- `src/hpvd/embedding.py`
- `inspect_scenario_a.py`
- `tests/test_synthetic_scenarios.py`
- `tests/test_contract.py`
- `tests/test_embedding.py`
- `tests/test_sparse_index.py`
- `tests/test_trajectory.py`
- Hapus direktori `synthetic_data/` jika ada.

*Penting:* File `src/hpvd/family.py` **JANGAN dihapus sepenuhnya**, tapi simplifikasi isinya. Sisakan HANYA 3 dataclass ini: `FamilyCoherence`, `StructuralSignature`, dan `UncertaintyFlags`. Sisanya (seperti `FamilyFormationEngine`, `AnalogFamily`, dll) boleh dihapus.

**Phase 2: Remove J-File Pipeline Layer**
Orchestration akan ditangani oleh Clara, jadi adapter J-file di HPVD dihapus:
- `src/hpvd/adapters/j_file_schemas.py`
- `src/hpvd/adapters/j13_adapter.py`
- `src/hpvd/adapters/j14_emitter.py`
- `src/hpvd/adapters/j15_emitter.py`
- `src/hpvd/adapters/j16_emitter.py`
- `src/hpvd/adapters/pipeline_engine.py`
- `src/hpvd/adapters/strategy_dispatcher.py`
- `src/hpvd/adapters/strategies/finance_strategy.py`

**Phase 3: Restructure + Cleanup**
Rapikan arsitektur menjadi 3 layer (api -> adapters -> infra):
1. Buat package `src/hpvd/infra/` (dengan `__init__.py` kosong).
2. Pindahkan `src/hpvd/kl_loader.py` menjadi `src/hpvd/infra/kl_loader.py`.
3. Pindahkan `src/hpvd/adapters/kl_client.py` menjadi `src/hpvd/infra/kl_client.py`.
4. Pindahkan file `src/hpvd/family.py` (yang sudah disimplifikasi) menjadi `src/hpvd/adapters/family_types.py`.
5. Update semua import path di file-file strategi (`knowledge_strategy.py`, `document_strategy.py`, `retrieval_strategy.py`) agar mengarah ke tempat yang baru (terutama untuk import dataclass dari `family_types.py`).
6. Bersihkan `src/hpvd/__init__.py`, `src/hpvd/adapters/__init__.py`, dan `src/hpvd/adapters/strategies/__init__.py` agar hanya mengekspor class yang masih ada.
7. Refaktor `src/hpvd/api.py`. Hapus penggunaan `HPVDPipelineEngine` dan J13. Di endpoint `/query`, langsung konversi payload masuk (yang berisi `sector` dan `observed`) menjadi dict query lalu panggil `KnowledgeRetrievalStrategy.search(query)`. Ubah format outputnya menjadi sesuai dengan yang dibutuhkan.
8. Bersihkan file test:
   - Di `tests/test_adapters.py`, hapus class test untuk Finance, J-file, Pipeline, dan Dispatcher. Sisakan test untuk antarmuka strategy dan DocumentStrategy.
   - Perbaiki import di `tests/test_kl_integration.py` dan `tests/test_knowledge_retrieval.py` agar sesuai dengan struktur baru.

Silakan mulai eksekusi dari Phase 1. Jika ada error, perbaiki satu per satu. Fokus pada hasil akhir di mana `api.py` dan sisa test bisa dijalankan tanpa error.
