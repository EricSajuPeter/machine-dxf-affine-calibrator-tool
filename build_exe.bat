@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ============================================================================
REM  AffineCalibratorGUI — Windows build (PyInstaller one-file EXE)
REM  Run from Explorer or cmd; creates .venv if missing, installs deps, builds.
REM
REM  Optional environment variables:
REM    SKIP_INSTALL=1     Skip pip install (use after deps already installed)
REM    PIP_UPGRADE=1      Pass --upgrade when installing requirements.txt
REM ============================================================================

set "APP_NAME=AffineCalibratorGUI"
set "VENV_PY=.venv\Scripts\python.exe"
set "SPEC_FILE=%APP_NAME%.spec"
set "EXE_OUT=dist\%APP_NAME%.exe"
set "BUILD_DIR=build"
set "DIST_DIR=dist"

REM Reduce Unicode/path quirks on Windows Python builds.
set "PYTHONUTF8=1"

echo ==========================================
echo   %APP_NAME% EXE Build Script
echo ==========================================
echo.

REM --- Preflight ---
if not exist "gui_app.py" (
  echo [ERROR] gui_app.py not found in: %cd%
  exit /b 1
)
if not exist "affine_core.py" (
  echo [ERROR] affine_core.py not found in: %cd%
  exit /b 1
)
if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found in: %cd%
  exit /b 1
)
if not exist "%SPEC_FILE%" (
  echo [ERROR] Spec file not found: %SPEC_FILE%
  exit /b 1
)

REM --- Virtual environment ---
if not exist ".venv" (
  echo [INFO] Creating virtual environment .venv ...
  where py >nul 2>&1
  if %errorlevel%==0 (
    py -3 -m venv .venv
  ) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
      python -m venv .venv
    ) else (
      echo [ERROR] Neither ^`py^` nor ^`python^` is on PATH. Install Python 3.10+ from python.org.
      exit /b 1
    )
  )
  if errorlevel 1 (
    echo [ERROR] Failed to create .venv
    exit /b 1
  )
)

if not exist "%VENV_PY%" (
  echo [ERROR] Virtualenv python not found: %cd%\%VENV_PY%
  exit /b 1
)

echo [INFO] Using: %cd%\%VENV_PY%
"%VENV_PY%" -c "import sys; print('[INFO] Python', sys.version.split()[0])"
if errorlevel 1 (
  echo [ERROR] Python in .venv failed to run.
  exit /b 1
)
echo.

REM --- Sanity check source before packaging ---
echo [INFO] Validating Python sources...
"%VENV_PY%" -m py_compile gui_app.py affine_core.py
if errorlevel 1 (
  echo [ERROR] Syntax check failed. Fix Python errors before building.
  exit /b 1
)
echo.

REM --- Dependencies ---
if /i "%SKIP_INSTALL%"=="1" (
  echo [INFO] SKIP_INSTALL=1 — skipping pip install.
) else (
  echo [INFO] Upgrading pip...
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    exit /b 1
  )

  echo [INFO] Installing dependencies from requirements.txt...
  if /i "%PIP_UPGRADE%"=="1" (
    "%VENV_PY%" -m pip install --upgrade -r requirements.txt
  ) else (
    "%VENV_PY%" -m pip install -r requirements.txt
  )
  if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    exit /b 1
  )
)
echo.

REM --- Clean previous PyInstaller output ---
echo [INFO] Building executable with PyInstaller...
if exist "%BUILD_DIR%" (
  echo [INFO] Removing previous "%BUILD_DIR%" ...
  rmdir /s /q "%BUILD_DIR%"
)
if exist "%DIST_DIR%" (
  echo [INFO] Removing previous "%DIST_DIR%" ...
  rmdir /s /q "%DIST_DIR%"
)

"%VENV_PY%" -m PyInstaller --noconfirm --clean --workpath "%BUILD_DIR%" --distpath "%DIST_DIR%" "%SPEC_FILE%"
if errorlevel 1 (
  echo [ERROR] PyInstaller build FAILED.
  exit /b 1
)

if exist "%EXE_OUT%" (
  echo.
  echo [OK] Build complete.
  echo [OK] Output: %cd%\%EXE_OUT%
  for %%A in ("%EXE_OUT%") do echo [OK] Size: %%~zA bytes
) else (
  echo.
  echo [WARN] Build finished but EXE not found at expected path:
  echo        %cd%\%EXE_OUT%
  echo [WARN] Check PyInstaller output above for the actual dist location.
  exit /b 1
)

echo.
echo [INFO] Done.
endlocal
exit /b 0
