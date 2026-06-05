param(
    [string]$Config = "configs/config_lugha_local.yaml",
    [string[]]$Splits = @("train", "val", "test")
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath "C:\Users\jdmay\lughallama-health_qa"

if (-not (Test-Path -LiteralPath ".venv")) {
    py -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

$env:PYTHONPATH = "$(Get-Location)\src"
& ".\.venv\Scripts\python.exe" src\data\clean_lugha_data.py --config $Config --splits $Splits

Write-Host ""
Write-Host "Cleaning complete. Review:"
Write-Host "  data\cleaned\train_clean_lugha.csv"
Write-Host "  data\cleaned\val_clean_lugha.csv"
Write-Host "  data\cleaned\test_clean_lugha.csv"
Write-Host "  reports\cleaning_report.json"
Write-Host "  reports\cleaning_issues.csv"
