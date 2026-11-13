$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$tasks = Join-Path $root "tasks"
$out   = Join-Path $root "out"
$logs  = Join-Path $root "logs"
foreach ($d in $tasks, $out, $logs) { New-Item -ItemType Directory -Force $d | Out-Null }

# 1. Docker must be running
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running. Start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# 2. Build image
Write-Host "== Building image hack:latest ==" -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()
docker build -t hack:latest .
$sw.Stop()
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed. See the error above." -ForegroundColor Red
    exit 1
}
Write-Host ("Image built in {0:N0} s ({1:mm\:ss})" -f $sw.Elapsed.TotalSeconds, $sw.Elapsed) -ForegroundColor Green

# 3. Choose mode
Write-Host ""
Write-Host "Modes:"
Write-Host "  1 - local only (phase 1, no LLM, 0 tokens)"
Write-Host "  2 - LLM for tasks that local solvers did not solve"
Write-Host "  3 - forced LLM (NO_LOCAL=1, every task goes to the model)"
$choice = Read-Host "Choose 1, 2 or 3 [1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }
if ($choice -notin @("1", "2", "3")) {
    Write-Host "Unknown choice: $choice" -ForegroundColor Red
    exit 1
}
$modeName = @{ "1" = "local"; "2" = "llm"; "3" = "forced-llm" }[$choice]
$useLlm   = ($choice -eq "2") -or ($choice -eq "3")

$extra = @()
if ($useLlm) {
    $env:LLM_BASE_URL = Read-Host "LLM_BASE_URL"
    $secure = Read-Host "LLM_API_KEY" -AsSecureString
    $env:LLM_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                           [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    if ([string]::IsNullOrWhiteSpace($env:LLM_API_KEY) -or [string]::IsNullOrWhiteSpace($env:LLM_BASE_URL)) {
        Write-Host "LLM_BASE_URL and LLM_API_KEY are required for this mode." -ForegroundColor Red
        exit 1
    }
    $budget = Read-Host "Token budget (Enter = no limit)"
    if (-not [string]::IsNullOrWhiteSpace($budget)) { $extra += @("--budget", $budget) }
    $extra += @("--workers", "3")
} else {
    $extra += "--local-only"
}

# 4. Run
$stamp    = Get-Date -Format "yyyyMMdd_HHmmss"
$trajName = "traj_$stamp.jsonl"
$log      = Join-Path $logs "run_${modeName}_$stamp.log"

$envArgs = @("-e", "HOME=/tmp", "-e", "TRAJ_FILE=/tmp/out/$trajName")
if ($useLlm)         { $envArgs += @("-e", "LLM_BASE_URL", "-e", "LLM_API_KEY") }
if ($choice -eq "3") { $envArgs += @("-e", "NO_LOCAL=1") }

$runArgs = @("run", "--rm", "--name", "hack-run") + $envArgs + @(
    "-v", "${tasks}:/data/tasks:rw",
    "-v", "${out}:/tmp/out:rw",
    "--read-only", "--tmpfs", "/tmp:rw,size=2g",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    "--cpus", "4", "--memory", "8g", "--pids-limit", "512",
    "--workdir", "/app", "--entrypoint", "python", "hack:latest",
    "/app/run_agent.py", "--tasks-dir", "/data/tasks",
    "--out", "/tmp/out/solutions.json"
) + $extra

Write-Host "== Running ($modeName), log: $log ==" -ForegroundColor Cyan
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& docker @runArgs 2>&1 | Tee-Object -FilePath $log
$exit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
Write-Host "Exit code: $exit"

# 5. Answers
Write-Host ""
Write-Host "== Answers ==" -ForegroundColor Cyan
Get-ChildItem $tasks -Directory | ForEach-Object {
    $a = Get-Content (Join-Path $_.FullName "answer.txt") -ErrorAction SilentlyContinue | Select-Object -First 1
    "{0,-22} {1}" -f $_.Name, $a
}

# 6. Tokens
Write-Host ""
Write-Host "== Tokens ==" -ForegroundColor Cyan
$doneLine = Get-Content $log | Select-String -Pattern "done: .*spent=([0-9.]+)" | Select-Object -Last 1
if ($doneLine) {
    $spent = $doneLine.Matches[0].Groups[1].Value
    Write-Host "Total spent (harness log): $spent output-token equivalents"
} else {
    Write-Host "Total spent: not found in log (the run may have failed)"
}

$trajPath = Join-Path $out $trajName
if (Test-Path $trajPath) {
    $rows = Get-Content $trajPath -Encoding utf8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
    $sum = ($rows | Measure-Object -Property cost -Sum).Sum
    Write-Host ("Sum of task costs (trajectories): {0}" -f $sum)
    $rows | Group-Object path | ForEach-Object {
        Write-Host ("  path={0}: {1} tasks" -f $_.Name, $_.Count)
    }
}

Write-Host ""
Write-Host "Solutions: $out\solutions.json"
Write-Host "Trajectories: $trajPath"