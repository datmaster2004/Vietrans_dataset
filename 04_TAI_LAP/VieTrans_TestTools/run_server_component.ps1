[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "probe",
        "ocr",
        "inpainting",
        "translation",
        "deployed-ocr",
        "deployed-inpainting"
    )]
    [string]$Mode,

    [string]$Manifest = "",
    [string]$OutputDir = "",
    [int]$Limit = 0,
    [ValidateRange(1, 8)]
    [int]$Workers = 3,
    [Alias("SpaceUrl")]
    [string]$ServerUrl = "",
    [Alias("SpaceId")]
    [string]$ServerId = "",
    [string]$BuildId = "",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$runner = Join-Path $root "tools\run_component_server.py"

if (-not $ServerUrl) {
    $ServerUrl = if ($env:VIETRANS_SERVER_URL) { $env:VIETRANS_SERVER_URL } else { "http://127.0.0.1:7860" }
}
if (-not $ServerId) {
    $ServerId = if ($env:VIETRANS_SERVER_ID) { $env:VIETRANS_SERVER_ID } else { "local-vietrans" }
}
if (-not $BuildId) {
    $BuildId = if ($env:VIETRANS_BUILD_ID) { $env:VIETRANS_BUILD_ID } else { "unknown" }
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing $python. Run .\bootstrap.ps1 first."
}

$arguments = @(
    $runner,
    $Mode,
    "--server-url", $ServerUrl,
    "--server-id", $ServerId,
    "--build-id", $BuildId
)
if ($Mode -ne "probe") {
    if (-not $Manifest) { throw "-Manifest is required for mode $Mode." }
    if (-not $OutputDir) { throw "-OutputDir is required for mode $Mode." }
    if (-not (Test-Path -LiteralPath $Manifest)) {
        throw "Manifest not found: $Manifest"
    }
    $arguments += @(
        "--manifest", $Manifest,
        "--output-dir", $OutputDir,
        "--workers", [string]$Workers
    )
    if ($Limit -gt 0) { $arguments += @("--limit", [string]$Limit) }
    if ($Resume) { $arguments += "--resume" }
} elseif ($OutputDir) {
    $arguments += @("--output-dir", $OutputDir)
}

& $python @arguments
exit $LASTEXITCODE
