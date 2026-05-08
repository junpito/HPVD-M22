$source = "D:\project\M22\HPVD-M22"
$dest = "D:\project\M22\HPVD-Manithy"

# Pastikan path absolut konsisten untuk regex
$sourceRegex = [regex]::Escape($source)

Write-Host "Creating new repository at $dest..." -ForegroundColor Cyan

# 1. Create directory
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# 2. Copy files (excluding .git, venv, pycache)
Write-Host "Copying files (this might take a few seconds)..." -ForegroundColor Cyan
Get-ChildItem -Path $source -Recurse | Where-Object {
    $path = $_.FullName
    -not ($path -match "\\\.git") -and
    -not ($path -match "\\venv") -and
    -not ($path -match "\\__pycache__") -and
    -not ($path -match "\\\.pytest_cache") -and
    -not ($path -match "migrate_repo\.ps1")
} | ForEach-Object {
    # Gunakan -replace (regex, case-insensitive)
    $targetPath = $_.FullName -replace $sourceRegex, $dest
    
    if ($_.PSIsContainer) {
        New-Item -ItemType Directory -Force -Path $targetPath | Out-Null
    } else {
        Copy-Item -Path $_.FullName -Destination $targetPath -Force
    }
}

# 3. Initialize Git and commit
Write-Host "Initializing Git repository..." -ForegroundColor Cyan
Set-Location -Path $dest
git init
git add .
git commit -m "Initial commit: Refactored HPVD knowledge & document retrieval engine"

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
Write-Host "Migration Complete!" -ForegroundColor Green
Write-Host "Your new repository is ready at: $dest" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. cd D:\project\M22\HPVD-Manithy"
Write-Host "2. python -m venv venv"
Write-Host "3. .\venv\Scripts\activate"
Write-Host "4. pip install -e ."
