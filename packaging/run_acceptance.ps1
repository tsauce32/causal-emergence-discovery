param(
    [string]$Python = "C:\Users\Owner\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$DependencyRoot = "C:\Users\Owner\Documents\ChatGPT\Causal Emergence\research\review_dependencies"
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ArtifactRoot = Join-Path $PSScriptRoot "artifacts"
$DistributionRoot = Join-Path $ArtifactRoot "dist"
$PytestTempRoot = Join-Path $ArtifactRoot "pytest-temp"
$LogPath = Join-Path $ArtifactRoot "package-acceptance.log"

New-Item -ItemType Directory -Force -Path $DistributionRoot | Out-Null
$env:PYTHONPATH = $DependencyRoot
$env:CED_PYTHON_EXE = $Python
$env:CED_PACKAGE_ACCEPTANCE_ARTIFACT_DIR = $DistributionRoot

Push-Location $Repository
try {
    & $Python -m build --no-isolation --sdist --wheel --outdir $DistributionRoot 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "Wheel/sdist build failed with exit code $LASTEXITCODE."
    }

    & $Python -m pytest -q --basetemp $PytestTempRoot tests/test_package_acceptance.py 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Installed-package acceptance failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Acceptance artifacts and command output: $ArtifactRoot"
