param(
    [string]$Config = "configs/config_lugha_local.yaml",
    [string[]]$Splits = @("train", "val", "test")
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath "C:\Users\jdmay\lughallama-health_qa"

if ($Splits.Count -eq 1 -and $Splits[0].Contains(",")) {
    $Splits = $Splits[0].Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath ".venv")) {
    Invoke-Native py -m venv .venv
}

Invoke-Native ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
Invoke-Native ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

$env:PYTHONPATH = "$(Get-Location)\src"
Invoke-Native ".\.venv\Scripts\python.exe" src\data\clean_lugha_data.py --config $Config --splits $Splits

$expected = @()
if ($Splits -contains "train") { $expected += "data\cleaned\train_clean_lugha.csv" }
if ($Splits -contains "val") { $expected += "data\cleaned\val_clean_lugha.csv" }
if ($Splits -contains "test") { $expected += "data\cleaned\test_clean_lugha.csv" }
$expected += "reports\cleaning_report.json"
$expected += "reports\cleaning_issues.csv"

$missing = $expected | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($missing.Count -gt 0) {
    throw "Cleaning finished but expected files are missing: $($missing -join ', ')"
}

Write-Host ""
Write-Host "Cleaning complete. Review:"
foreach ($file in $expected) {
    Write-Host "  $file"
}
