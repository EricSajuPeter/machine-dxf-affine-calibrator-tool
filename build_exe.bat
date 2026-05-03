@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "SPEC_FILE=AffineCalibratorGUI.spec"
set "EXE_OUT=dist\AffineCalibratorGUI.exe"
set "BUILD_DIR=build"
set "DIST_DIR=dist"

echo ==========================================
echo   AffineCalibratorGUI EXE Build Script
echo ==========================================
echo.

if not exist ".venv" (
  echo [INFO] Creating virtual environment...
  where py >nul 2>&1
  if %errorlevel%==0 (
    py -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 (
    echo [ERROR] Failed to create .venv
    exit /b 1
  )
)

if not exist "%VENV_PY%" (
  echo [ERROR] Virtualenv python not found: %VENV_PY%
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

echo [INFO] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] pip upgrade failed.
  exit /b 1
)

echo [INFO] Installing dependencies...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Dependency install failed.
  exit /b 1
)

echo [INFO] Building executable with PyInstaller...
if exist "%BUILD_DIR%" (
  echo [INFO] Removing previous build folder...
  rmdir /s /q "%BUILD_DIR%"
)
if exist "%DIST_DIR%" (
  echo [INFO] Removing previous dist folder...
  rmdir /s /q "%DIST_DIR%"
)

"%VENV_PY%" -m PyInstaller --noconfirm --clean --workpath "%BUILD_DIR%" --distpath "%DIST_DIR%" "%SPEC_FILE%"
if errorlevel 1 (
  echo [ERROR] Build FAILED.
  exit /b 1
)

if exist "%EXE_OUT%" (
  echo.
  echo [OK] Build complete.
  echo [OK] Output: %cd%\%EXE_OUT%
) else (
  echo.
  echo [WARN] Build finished but EXE not found at expected path:
  echo        %cd%\%EXE_OUT%
  echo [WARN] Check PyInstaller output above for details.
  exit /b 1
)

echo.
echo [INFO] Done.
endlocal
