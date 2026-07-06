# GitHub Push Script for triton-kernel-zoo
# Usage: Just double-click this file or run: .\push.ps1

$ErrorActionPreference = "Stop"
$REPO = "c:\Users\gaofe\Desktop\dive-into-llm\AI Infra\triton-kernel-zoo"
$GH = "C:\Program Files\GitHub CLI\gh.exe"

function Get-Commit-Message {
    $templates = @(
        "feat: add new Triton kernel implementation",
        "feat: update benchmark results",
        "feat: optimize kernel performance",
        "feat: add kernel documentation",
        "feat: improve kernel test coverage",
        "refactor: clean up kernel code",
        "fix: resolve kernel correctness issue",
        "docs: update README and documentation",
        "chore: update project dependencies"
    )
    $idx = Get-Random -Minimum 0 -Maximum $templates.Count
    return $templates[$idx]
}

Set-Location $REPO

Write-Host ""
Write-Host "=== GitHub Push for triton-kernel-zoo ===" -ForegroundColor Cyan
Write-Host ""

# Check git status
$status = git status --porcelain 2>&1
if ($null -eq $status -or $status -eq "") {
    Write-Host "[INFO] No changes to commit." -ForegroundColor Yellow
    exit 0
}

Write-Host "Changes detected:" -ForegroundColor Green
Write-Host $status
Write-Host ""

# Stage all changes
git add -A
Write-Host "[OK] All changes staged." -ForegroundColor Green

# Commit with random message
$msg = Get-Commit-Message
Write-Host "Commit message: $msg" -ForegroundColor Gray
git commit -m $msg
Write-Host "[OK] Changes committed." -ForegroundColor Green

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Gray
git push origin main 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[SUCCESS] Pushed to https://github.com/Gf0205/triton-kernel-zoo" -ForegroundColor Green
    Write-Host ""
    Write-Host "View your repo: https://github.com/Gf0205/triton-kernel-zoo" -ForegroundColor Cyan
} else {
    Write-Host "[ERROR] Push failed. Check your network or authentication." -ForegroundColor Red
}

Write-Host ""
Write-Host "Press Enter to exit..."
Read-Host
