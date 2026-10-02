param(
    [ValidateSet('quick','full')][string]$Profile = 'full',
    [ValidateRange(1,64)][int]$Workers = 4,
    [string]$PythonPath = 'python',
    [string]$GitPath = 'git',
    [string]$DependenciesPath = '',
    [string]$SourcePath = '',
    [string]$StatusPath = '',
    [string]$OutputPath = ''
)
$ErrorActionPreference = 'Stop'
$taskCheckout = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$taskWorkspace = (Resolve-Path -LiteralPath (Join-Path $taskCheckout '..\..')).Path
if (-not $SourcePath) { $SourcePath = Join-Path $taskWorkspace 'work\discovery-integrated' }
if (-not $StatusPath) { $StatusPath = Join-Path $taskWorkspace 'research\discovery-integrated\status.json' }
if (-not $OutputPath) { $OutputPath = Join-Path $taskWorkspace ('research\discovery-pipeline-benchmark\' + $Profile) }
$taskStatus = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
if (-not $taskStatus.ready_for_integration -or -not $taskStatus.commit) {
    throw 'The integrated source must be ready_for_integration=true with a pinned commit.'
}
foreach ($taskArtifact in @('manifest.json', 'trials.jsonl', 'summary.json')) {
    if (Test-Path -LiteralPath (Join-Path $OutputPath $taskArtifact)) {
        throw "Preserve existing experiment evidence; choose a fresh OutputPath ($taskArtifact already exists)."
    }
}
$taskGitResolved = (Get-Command $GitPath -ErrorAction Stop).Source
$env:PATH = (Split-Path -Parent $taskGitResolved) + [IO.Path]::PathSeparator + $env:PATH
$taskPythonPaths = @($taskCheckout)
if ($DependenciesPath) { $taskPythonPaths += (Resolve-Path -LiteralPath $DependenciesPath).Path }
$env:PYTHONPATH = $taskPythonPaths -join [IO.Path]::PathSeparator
Push-Location -LiteralPath $taskCheckout
try {
    & $PythonPath -m benchmarks.pipeline.runner --profile $Profile --workers $Workers --source $SourcePath --expected-commit $taskStatus.commit --readiness-status $StatusPath --output $OutputPath
    if ($LASTEXITCODE -ne 0) { throw "Benchmark exited with status $LASTEXITCODE" }
} finally {
    Pop-Location
}
