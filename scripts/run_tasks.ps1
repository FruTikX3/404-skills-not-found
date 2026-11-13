# Runs the agent on every task in .\tasks.
# Each run works on a fresh copy of the tasks, so older runs never leak into results.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$src = Join-Path $root "tasks"
$taskDirs = @()
foreach ($d in (Get-ChildItem $src -Directory -ErrorAction SilentlyContinue)) {
    if (Test-Path (Join-Path $d.FullName "task.json")) { $taskDirs += $d }
}
if ($taskDirs.Count -eq 0) {
    Write-Host "[ERROR] No task folders with task.json found in $src" -ForegroundColor Red
    exit 1
}
Write-Host ("Found {0} tasks in {1}" -f $taskDirs.Count, $src) -ForegroundColor Cyan

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$run   = Join-Path $root ("runs\" + $stamp)
$tasks = Join-Path $run "tasks"
$out   = Join-Path $run "out"
New-Item -ItemType Directory -Force $tasks | Out-Null
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item -Path (Join-Path $src "*") -Destination $tasks -Recurse -Force
# drop answers left over from older runs in the working copy
Get-ChildItem $tasks -Recurse -Filter answer.txt | Remove-Item -Force
Write-Host "Working copy: $tasks"

$Mode = "llm"

$envArgs = @()
$envArgs += "-e"
$envArgs += "HOME=/tmp"
$envArgs += "-e"
$envArgs += "TRAJ_FILE=/tmp/out/traj_$stamp.jsonl"

$agentArgs = @()
$agentArgs += "--tasks-dir"
$agentArgs += "/data/tasks"
$agentArgs += "--out"
$agentArgs += "/tmp/out/solutions.json"

$budgetInput = Read-Host "Token budget [default 100000]"
if ([string]::IsNullOrWhiteSpace($budgetInput)) { $budget = "100000" } else { $budget = $budgetInput.Trim() }
$agentArgs += "--budget"
$agentArgs += $budget
$agentArgs += "--workers"
$agentArgs += "2"

if ($env:LLM_BASE_URL) { $envArgs += "-e"; $envArgs += "LLM_BASE_URL" }
if ($env:LLM_API_KEY)  { $envArgs += "-e"; $envArgs += "LLM_API_KEY" }

$dockerArgs = @()
$dockerArgs += "run"
$dockerArgs += "--rm"
$dockerArgs += "--name"
$dockerArgs += "hack-run-$stamp"
$dockerArgs += $envArgs
$dockerArgs += "-v"
$dockerArgs += "${tasks}:/data/tasks:rw"
$dockerArgs += "-v"
$dockerArgs += "${out}:/tmp/out:rw"
$dockerArgs += "--read-only"
$dockerArgs += "--tmpfs"
$dockerArgs += "/tmp:rw,size=2g"
$dockerArgs += "--cap-drop"
$dockerArgs += "ALL"
$dockerArgs += "--security-opt"
$dockerArgs += "no-new-privileges"
$dockerArgs += "--cpus"
$dockerArgs += "4"
$dockerArgs += "--memory"
$dockerArgs += "8g"
$dockerArgs += "--pids-limit"
$dockerArgs += "512"
$dockerArgs += "--workdir"
$dockerArgs += "/app"
$dockerArgs += "--entrypoint"
$dockerArgs += "python"
$dockerArgs += "hack:latest"
$dockerArgs += "/app/run_agent.py"
$dockerArgs += $agentArgs

$log = Join-Path $run "run.log"
Write-Host "== Running ($Mode) ==" -ForegroundColor Cyan
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& docker @dockerArgs 2>&1 | ForEach-Object {
    Write-Host $_
    Add-Content -Path $log -Value $_ -Encoding utf8
}
$exit = $LASTEXITCODE
$ErrorActionPreference = $prev

Write-Host ""
Write-Host "== Answers ==" -ForegroundColor Cyan
$answered = 0
foreach ($d in $taskDirs) {
    $dir = Join-Path $tasks $d.Name
    $ans = Join-Path $dir "answer.txt"
    if (-not (Test-Path $ans)) { $ans = Join-Path $dir "workspace\answer.txt" }
    if (Test-Path $ans) {
        $answered++
        $first = Get-Content $ans -Encoding utf8 -TotalCount 1
        Write-Host ("{0,-24} {1}" -f $d.Name, $first)
    } else {
        Write-Host ("{0,-24} -" -f $d.Name) -ForegroundColor DarkGray
    }
}

Write-Host ""
$done = Select-String -Path $log -Pattern "done: .*spent=" | Select-Object -Last 1
if ($done) { Write-Host ("Result: " + ($done.Line -replace '^.*done: ', 'done: ')) }
Write-Host ("Answered: {0} of {1}" -f $answered, $taskDirs.Count)
Write-Host "Solutions:    $out\solutions.json"
Write-Host "Run log:      $log"
exit $exit