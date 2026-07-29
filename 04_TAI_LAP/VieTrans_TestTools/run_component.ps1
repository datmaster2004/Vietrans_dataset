[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("ocr", "inpainting", "translation")]
    [string]$Suite,

    [Parameter(Mandatory = $true)]
    [string]$Manifest,

    [Parameter(Mandatory = $true)]
    [string]$Predictions,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [ValidateRange(0.01, 1.0)]
    [double]$IouThreshold = 0.5
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$evaluator = Join-Path $root "tools\evaluate.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing $python. Run .\bootstrap.ps1 first."
}
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Manifest not found: $Manifest"
}
if (-not (Test-Path -LiteralPath $Predictions)) {
    throw "Predictions not found: $Predictions"
}

$arguments = @(
    $evaluator,
    $Suite,
    "--manifest", $Manifest,
    "--predictions", $Predictions,
    "--output-dir", $OutputDir
)
if ($Suite -eq "ocr") {
    $arguments += @("--iou-threshold", [string]$IouThreshold)
}

& $python @arguments
exit $LASTEXITCODE

