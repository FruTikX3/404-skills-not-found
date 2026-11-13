@echo off
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
pushd "%ROOT%"
set "ERR=0"
set "DD_EXE="
set "DEPS_TRIED=0"

echo ============================================
echo  Offline start: nothing is downloaded
echo ============================================

rem ---- 1. Docker: check whether it is running; if not, find Docker Desktop and start it
:check_docker
where docker >nul 2>&1
if errorlevel 1 goto :no_docker
docker info >nul 2>&1
if not errorlevel 1 goto :docker_ready

echo [INFO] Docker is installed but the engine is NOT running now.
for /f "delims=" %%P in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\find_docker.ps1"') do set "DD_EXE=%%P"
if not defined DD_EXE goto :docker_no_exe
echo [INFO] Found Docker Desktop: "%DD_EXE%"
echo [INFO] Starting Docker Desktop...
start "" "%DD_EXE%"
set /a WAITED=0
:docker_wait
timeout /t 5 /nobreak >nul
docker info >nul 2>&1
if not errorlevel 1 goto :docker_ready
set /a WAITED+=5
if %WAITED% GEQ 300 goto :docker_timeout
echo        waiting for Docker engine... %WAITED% s
goto :docker_wait

:docker_ready
set "DOCKOS="
for /f "delims=" %%O in ('docker info --format "{{.OSType}}" 2^>nul') do set "DOCKOS=%%O"
if /i not "%DOCKOS%"=="linux" goto :docker_mode
echo [OK] Docker engine is RUNNING now, mode: %DOCKOS%
goto :check_python

:no_docker
echo [ERROR] Docker is not installed on this PC. Install Docker Desktop from the prepared package and run this file again.
set "ERR=1"
goto :summary
:docker_no_exe
echo [ERROR] Docker Desktop.exe was not found and the engine is not running.
echo         Start Docker Desktop by hand and run this file again.
set "ERR=1"
goto :summary
:docker_timeout
echo [ERROR] Docker did not become ready within 5 minutes. Check Docker Desktop and try again.
set "ERR=1"
goto :summary
:docker_mode
echo [ERROR] Docker runs in "%DOCKOS%" mode. Switch to Linux containers (tray icon, "Switch to Linux containers").
set "ERR=1"
goto :summary

rem ---- 2. Python 3.10+ (3.11 recommended)
:check_python
where python >nul 2>&1
if errorlevel 1 goto :no_python
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto :old_python
for /f "delims=" %%V in ('python -c "import sys; print(sys.version.split()[0])"') do set "PYV=%%V"
echo [OK] Python %PYV%
echo %PYV%| findstr /b "3.11." >nul || echo [WARN] Python 3.11 recommended, found %PYV%
goto :check_files
:no_python
echo [ERROR] Python is not installed. Install Python 3.11 from the prepared package and run this file again.
set "ERR=1"
goto :summary
:old_python
echo [ERROR] Python 3.10 or newer is required.
set "ERR=1"
goto :summary

rem ---- 3. Project files
:check_files
for %%F in (Dockerfile run_agent.py requirements.txt .dockerignore scripts\check_deps.py scripts\run_tasks.ps1 scripts\find_docker.ps1) do (
    if not exist "%%F" (
        echo [ERROR] Missing file: %%F
        set "ERR=1"
    )
)
if "%ERR%"=="1" goto :summary

rem ---- 4. Host Python packages are NOT needed to run the tasks: the agent and its packages live inside the image.
python "%ROOT%scripts\check_deps.py" "%ROOT%requirements.txt" >nul 2>&1
if errorlevel 1 echo [INFO] Host Python packages are not checked here. This is fine: the tasks run inside the Docker image.

rem ---- 5. Docker image hack:latest: load from the prepared archive (no build, no pull)
:check_image
docker image inspect hack:latest >nul 2>&1
if not errorlevel 1 goto :image_ok
if not exist "%ROOT%images\hack_all.tar" goto :no_image
echo [INFO] Loading the image from images\hack_all.tar ...
docker load -i "%ROOT%images\hack_all.tar"
docker image inspect hack:latest >nul 2>&1
if errorlevel 1 goto :no_image
:image_ok
echo [OK] Image hack:latest is available locally.
goto :summary
:no_image
echo [ERROR] Image hack:latest is missing. Copy images\hack_all.tar next to this file (it is created on the development PC).
set "ERR=1"

rem ---- 6. Run the tasks
:summary
echo.
if "%ERR%"=="1" (
    echo Prerequisites are NOT satisfied. See the messages above.
    popd
    pause
    exit /b 1
)
echo All prerequisites satisfied. Running the tasks...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\run_tasks.ps1"
set "RC=%ERRORLEVEL%"
popd
pause
exit /b %RC%
