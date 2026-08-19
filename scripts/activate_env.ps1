[CmdletBinding()]
param(
    [string]$EnvPath = '',
    [string]$EnvName = 'robot-olp-toolkit'
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw 'conda is not available in the current PowerShell session.'
}

$condaBase = (& conda info --base).Trim()
$hookPath = Join-Path $condaBase 'shell\condabin\conda-hook.ps1'
if (-not (Test-Path -LiteralPath $hookPath)) {
    throw "Conda PowerShell hook not found: $hookPath"
}

if (-not $EnvPath) {
    $localEnv = Join-Path $projectRoot 'env'
    $namedEnv = Join-Path (Join-Path $condaBase 'envs') $EnvName
    if (Test-Path -LiteralPath (Join-Path $localEnv 'python.exe')) {
        $EnvPath = $localEnv
    } elseif (Test-Path -LiteralPath (Join-Path $namedEnv 'python.exe')) {
        $EnvPath = $namedEnv
    } else {
        throw "No runnable project environment found. Checked: $localEnv and $namedEnv"
    }
}

$resolvedEnv = (Resolve-Path -LiteralPath $EnvPath).Path
. $hookPath
conda activate $resolvedEnv

python -c "import PySide6, pyvista, scipy, toppra, ruckig; print('Runtime dependencies: OK')"
if ($LASTEXITCODE -ne 0) {
    throw 'The environment is active, but required runtime dependencies are incomplete.'
}

Write-Host "Activated environment: $resolvedEnv" -ForegroundColor Green
Write-Host 'Run: python -m main_app.main_app' -ForegroundColor Cyan
