# Focused integration checks for output archiving and fail-fast native commands.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File tests/windows_runner_checks.ps1
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$parseErrors = $null
$parseTokens = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $repositoryRoot 'scripts/windows_runner.ps1'), [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$functions = $ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $false)
foreach ($function in $functions) {
    . ([scriptblock]::Create($function.Extent.Text))
}

# Use an isolated fixture, never the real research outputs.
$projectRoot = Join-Path $repositoryRoot ('.runtime/launcher-test-' + [Guid]::NewGuid().ToString('N'))
$runId = 'fixture'
New-Item -ItemType Directory -Path (Join-Path $projectRoot 'results/full') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $projectRoot 'artifacts/final') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $projectRoot 'data/raw') -Force | Out-Null
Set-Content -LiteralPath (Join-Path $projectRoot 'results/full/summary.json') -Value 'previous results'
Set-Content -LiteralPath (Join-Path $projectRoot 'artifacts/final/j48.model') -Value 'previous model'
Set-Content -LiteralPath (Join-Path $projectRoot 'data/raw/source.arff') -Value 'source untouched'
Archive-Outputs
if (Test-Path -LiteralPath (Join-Path $projectRoot 'results/full')) { throw 'Old results were not archived.' }
if ((Get-Content -LiteralPath (Join-Path $projectRoot '.runtime/backups/fixture/results/full/summary.json')) -ne 'previous results') {
    throw 'Archived results did not preserve their contents.'
}
if ((Get-Content -LiteralPath (Join-Path $projectRoot '.runtime/backups/fixture/artifacts/final/j48.model')) -ne 'previous model') {
    throw 'Archived model did not preserve its contents.'
}
if ((Get-Content -LiteralPath (Join-Path $projectRoot 'data/raw/source.arff')) -ne 'source untouched') { throw 'Source was modified.' }
$rejectedOutside = $false
try { Assert-WorkspacePath (Join-Path $projectRoot '../outside') | Out-Null } catch { $rejectedOutside = $true }
if (-not $rejectedOutside) { throw 'An outside-workspace archive path was accepted.' }
$rejectedCommand = $false
try { Invoke-Checked 'cmd.exe' @('/c', 'exit', '7') } catch { $rejectedCommand = $true }
if (-not $rejectedCommand) { throw 'A failed command did not stop the workflow.' }
Write-Host 'PASS: archives preserve previous outputs and source data; outside paths and failed commands are rejected.'
Write-Host "Fixture retained for inspection: $projectRoot"
