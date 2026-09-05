[CmdletBinding()]
param(
    [ValidateSet('Start', 'FullRun')][string]$Mode = 'Start',
    [ValidateRange(1024, 65535)][int]$Port = 8501,
    [switch]$CheckOnly,
    [switch]$NoBrowser,
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
Set-Location -LiteralPath $projectRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$runId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$logDirectory = Join-Path $projectRoot '.runtime/logs'
$transcriptStarted = $false
$resultCode = 0

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Program $($Arguments -join ' ')"
    }
}

function Test-Python {
    param([string]$Program, [string[]]$Prefix = @())
    if (-not $Program) { return $false }
    try {
        & $Program @Prefix -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)' *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Ensure-Python {
    if (Test-Python $pythonPath) { return }
    if ($Mode -eq 'Start' -or $CheckOnly) {
        throw 'A working .venv with 64-bit Python 3.12 is required. Install Python 3.12 and run FullRun.bat.'
    }
    if (Test-Path -LiteralPath (Join-Path $projectRoot '.venv')) {
        throw 'The existing .venv is incompatible or broken. Rename it, then rerun FullRun.bat to create a fresh environment.'
    }
    $basePython = $null
    $prefix = @()
    foreach ($candidate in @(Get-ChildItem -Path "$projectRoot/.runtime/python/*/python.exe" -ErrorAction SilentlyContinue)) {
        if (Test-Python $candidate.FullName) { $basePython = $candidate.FullName; break }
    }
    if (-not $basePython) {
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($launcher -and (Test-Python $launcher.Source @('-3.12'))) {
            $basePython = $launcher.Source
            $prefix = @('-3.12')
        }
    }
    if (-not $basePython) {
        $command = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($command -and (Test-Python $command.Source)) { $basePython = $command.Source }
    }
    if (-not $basePython) { throw 'Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/ and rerun FullRun.bat.' }
    Invoke-Checked $basePython ($prefix + @('-m', 'venv', '.venv'))
    if (-not (Test-Python $pythonPath)) { throw 'The new Python environment could not be started.' }
}

function Ensure-Java {
    $homes = @()
    if ($env:JAVA_HOME) { $homes += $env:JAVA_HOME }
    $homes += @(Get-ChildItem -Path "$projectRoot/.runtime/java/*" -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    foreach ($homePath in $homes) {
        if ((Test-Path -LiteralPath (Join-Path $homePath 'bin/server/jvm.dll')) -and
            (Test-Path -LiteralPath (Join-Path $homePath 'bin/java.exe'))) {
            $env:JAVA_HOME = $homePath
            Invoke-Checked (Join-Path $homePath 'bin/java.exe') @('-version')
            return
        }
    }
    if ($Mode -eq 'Start' -or $CheckOnly) {
        throw 'Java was not found. Set JAVA_HOME to a compatible 64-bit Java installation or run FullRun.bat to download a local Java 17 runtime.'
    }
    Write-Host 'Downloading a project-local Eclipse Temurin Java 17 runtime...'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $release = Invoke-RestMethod -Uri 'https://api.adoptium.net/v3/assets/latest/17/hotspot?architecture=x64&image_type=jre&os=windows&vendor=eclipse'
    $package = @($release)[0].binary.package
    $archive = Join-Path $projectRoot '.runtime/java-download.zip'
    Invoke-WebRequest -UseBasicParsing -Uri $package.link -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $package.checksum) {
        throw 'Downloaded Java archive failed its SHA-256 check.'
    }
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $projectRoot '.runtime/java') -Force
    $javaHome = Get-ChildItem -Path "$projectRoot/.runtime/java/*" -Directory | Where-Object {
        Test-Path -LiteralPath (Join-Path $_.FullName 'bin/server/jvm.dll')
    } | Select-Object -First 1
    if (-not $javaHome) { throw 'Java archive did not contain the expected Windows runtime.' }
    $env:JAVA_HOME = $javaHome.FullName
    Invoke-Checked (Join-Path $env:JAVA_HOME 'bin/java.exe') @('-version')
}

function Assert-WorkspacePath {
    param([string]$Path)
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to archive a path outside this workspace: $absolute"
    }
    $cursor = $absolute
    while ($cursor -ne $projectRoot) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing to move outputs through a junction or symlink: $cursor"
            }
        }
        $cursor = Split-Path -Parent $cursor
    }
    return $absolute
}

function Archive-Outputs {
    $backupRoot = Assert-WorkspacePath (Join-Path $projectRoot ".runtime/backups/$runId")
    foreach ($relative in @('results/full', 'artifacts/final', 'docs/IMPLEMENTATION_RESULTS.md', 'docs/screenshots')) {
        $source = Assert-WorkspacePath (Join-Path $projectRoot $relative)
        if (Test-Path -LiteralPath $source) {
            $destination = Assert-WorkspacePath (Join-Path $backupRoot $relative)
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
            Move-Item -LiteralPath $source -Destination $destination
        }
    }
    Write-Host "Previous generated outputs preserved in $backupRoot"
}

function Find-AvailablePort {
    for ($candidate = $Port; $candidate -le [Math]::Min($Port + 9, 65535); $candidate++) {
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $candidate)
        try { $listener.Start(); return $candidate }
        catch [Net.Sockets.SocketException] { }
        finally { $listener.Stop() }
    }
    throw "No available local port in the range starting at $Port. Use -Port with another port number."
}

try {
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    Start-Transcript -Path (Join-Path $logDirectory "$Mode-$runId.log") | Out-Null
    $transcriptStarted = $true
    Write-Host "Pima C4.5 - $Mode"
    Write-Host "Project: $projectRoot"
    Ensure-Python
    Ensure-Java

    if ($Mode -eq 'FullRun' -and -not $CheckOnly) {
        Write-Host '[1/8] Installing and checking locked Python dependencies...'
        Invoke-Checked $pythonPath @('-m', 'ensurepip', '--upgrade')
        Invoke-Checked $pythonPath @('-m', 'pip', 'install', '-r', 'requirements.lock')
        Invoke-Checked $pythonPath @('-m', 'pip', 'install', '--no-deps', '-e', '.')
        Invoke-Checked $pythonPath @('-m', 'pip', 'check')
    }

    $preflight = @('scripts/check_environment.py')
    if ($Mode -eq 'Start') { $preflight += '--model' }
    Invoke-Checked $pythonPath $preflight
    $selectedPort = Find-AvailablePort
    if ($CheckOnly) {
        Write-Host "Checks passed. Available port: $selectedPort. No training or frontend launch was performed."
        if ($Mode -eq 'FullRun') {
            Write-Host 'FullRun sequence: locked dependencies -> preflight -> source conversion -> archive previous outputs -> core tests -> full nested evaluation -> fresh final training -> report -> all tests -> model verification -> Streamlit.'
        }
    } else {
        if ($Mode -eq 'FullRun') {
            Write-Host '[2/8] Verifying the source and recreating the dataset...'
            Invoke-Checked $pythonPath @('scripts/prepare_data.py')
            Invoke-Checked 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'tests/windows_runner_checks.ps1')
            Archive-Outputs
            Write-Host '[3/8] Testing the data pipeline and J48 integration...'
            Invoke-Checked $pythonPath @('-m', 'pytest', '-q', 'tests/test_pipeline.py')
            Write-Host '[4/8] Full ten-fold outer / five-fold inner evaluation. This can take a while...'
            Invoke-Checked $pythonPath @('-m', 'diabetes_c45.evaluate')
            Write-Host '[5/8] Fresh parameter search and final model training...'
            Invoke-Checked $pythonPath @('-m', 'diabetes_c45.train')
            Write-Host '[6/8] Generating research figures and the results report...'
            Invoke-Checked $pythonPath @('scripts/research_report.py')
            Write-Host '[7/8] Running all tests and verifying the saved model...'
            Invoke-Checked $pythonPath @('-m', 'pytest', '-q')
            Invoke-Checked $pythonPath @('scripts/check_environment.py', '--model')
            Write-Host '[8/8] Starting the frontend prototype...'
        }
        Write-Host "Prototype: http://127.0.0.1:$selectedPort"
        Write-Host 'Keep this window open. Press Ctrl+C to stop Streamlit.'
        $headless = if ($NoBrowser) { 'true' } else { 'false' }
        Invoke-Checked $pythonPath @('-m', 'streamlit', 'run', 'app/streamlit_app.py',
            '--server.address=127.0.0.1', "--server.port=$selectedPort",
            "--server.headless=$headless", '--browser.gatherUsageStats=false')
    }
} catch {
    $resultCode = 1
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Review the log in $logDirectory. Training and launch stop at the failed step."
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
if (-not $NoPause) { Read-Host 'Press Enter to close this window' | Out-Null }
exit $resultCode
