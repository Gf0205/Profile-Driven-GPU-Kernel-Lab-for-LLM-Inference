# Quick GitHub Update Script for triton-kernel-zoo
# Run from any terminal:  .\quick-push.ps1

$ErrorActionPreference = "Stop"
$REPO = "c:\Users\gaofe\Desktop\dive-into-llm\AI Infra\triton-kernel-zoo"

Set-Location $REPO

$status = git status --porcelain 2>&1
if ($null -eq $status -or $status -eq "") {
    Write-Host "[INFO] No changes to commit." -ForegroundColor Yellow
    exit 0
}

Write-Host "Changes detected:" -ForegroundColor Green
Write-Host $status
Write-Host ""

$msg = if ($args[0]) { $args[0] } else { "feat: update triton-kernel-zoo" }

git add -A
git commit -m $msg
git push origin main

Write-Host ""
Write-Host "[DONE] https://github.com/Gf0205/triton-kernel-zoo" -ForegroundColor Cyan
