# Panduan Upload HPVD-M22 ke GitHub

## Langkah-langkah

### 1. Inisialisasi Git Repository

```bash
cd d:\project\M22\HPVD-M22
git init
```

### 2. Tambahkan Semua File

```bash
git add .
```

### 3. Buat Commit Pertama

```bash
git commit -m "Initial commit: HPVD-M22 project"
```

### 4. Buat Repository di GitHub

1. Buka https://github.com
2. Klik tombol **"+"** di kanan atas → **"New repository"**
3. Isi:
   - **Repository name:** `HPVD-M22` (atau nama lain yang Anda inginkan)
   - **Description:** `Hybrid Probabilistic Vector Database for Trajectory Intelligence - Matrix22 Project`
   - **Visibility:** Public atau Private (sesuai kebutuhan)
   - **JANGAN** centang "Initialize with README" (karena kita sudah punya)
4. Klik **"Create repository"**

### 5. Tambahkan Remote dan Push

Setelah repository dibuat di GitHub, jalankan:

```bash
# Ganti YOUR_USERNAME dengan username GitHub Anda
git remote add origin https://github.com/YOUR_USERNAME/HPVD-M22.git

# Push ke GitHub
git branch -M main
git push -u origin main
```

### 6. Jika Sudah Ada Repository di GitHub

Jika repository sudah ada, cukup jalankan:

```bash
git remote add origin https://github.com/YOUR_USERNAME/HPVD-M22.git
git branch -M main
git push -u origin main
```

## Troubleshooting

### Jika ada error "remote origin already exists"

```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/HPVD-M22.git
```

### Jika perlu update credentials

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Jika push ditolak karena branch berbeda

```bash
git push -u origin main --force
```

**PERHATIAN:** `--force` hanya gunakan jika yakin, karena akan overwrite history di GitHub.

## File yang Diabaikan (.gitignore)

File/folder berikut **TIDAK** akan di-upload:
- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- `.pytest_cache/` - Test cache
- `.cursor/` - Cursor IDE files
- `*.pkl`, `*.npy`, `*.faiss` - Data files
- `*.log` - Log files
- `*.bak` - Backup files

## Verifikasi

Setelah push berhasil, cek di browser:
```
https://github.com/YOUR_USERNAME/HPVD-M22
```

Anda seharusnya melihat semua file project sudah ada di GitHub!

