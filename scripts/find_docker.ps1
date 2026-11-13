# Prints the full path of "Docker Desktop.exe" if it can be found by any of several methods, prints nothing otherwise.
# Methods: uninstall registry entries, Start Menu shortcuts, default install folders.
$candidates = New-Object System.Collections.ArrayList

$keys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
)
foreach ($e in (Get-ItemProperty -Path $keys -ErrorAction SilentlyContinue)) {
    if ($e.DisplayName -like "Docker Desktop*") {
        if ($e.InstallLocation) { [void]$candidates.Add((Join-Path $e.InstallLocation "Docker Desktop.exe")) }
        if ($e.DisplayIcon) { [void]$candidates.Add(($e.DisplayIcon -replace ',\s*\d+$', '' -replace '"', '')) }
    }
}

$shell = New-Object -ComObject WScript.Shell
$menus = @("$env:ProgramData\Microsoft\Windows\Start Menu\Programs", "$env:APPDATA\Microsoft\Windows\Start Menu\Programs")
foreach ($m in $menus) {
    Get-ChildItem -Path $m -Recurse -Filter "Docker Desktop*.lnk" -ErrorAction SilentlyContinue | ForEach-Object {
        [void]$candidates.Add($shell.CreateShortcut($_.FullName).TargetPath)
    }
}

[void]$candidates.Add((Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"))
[void]$candidates.Add((Join-Path $env:LOCALAPPDATA "Programs\Docker\Docker\Docker Desktop.exe"))

$hit = $candidates | Where-Object { $_ -and $_.ToLower().EndsWith(".exe") -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if ($hit) { Write-Output $hit }
