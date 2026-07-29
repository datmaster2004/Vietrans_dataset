[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"
$requirements = Join-Path $root "requirements.txt"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating isolated environment: $venv"
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
& $venvPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Could not install evaluation dependencies." }

Write-Host "Ready: $venvPython"
& $venvPython -c "import gradio_client, httpx, huggingface_hub, numpy, PIL, pyarrow, sacrebleu, skimage; print('dependency_check=OK')"
