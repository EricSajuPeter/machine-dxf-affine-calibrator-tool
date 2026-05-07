@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ============================================================================
REM  AffineCalibratorGUI — Windows build (PyInstaller one-file EXE)
REM  Run from Explorer or cmd; creates .venv if missing, installs deps, builds.
REM
REM  Changelog (latest):
REM    - Embedded app icon assets into the EXE bundle and improved runtime taskbar icon reliability on Windows.
REM    - Build now requires icon asset in assets folder and auto-generates app_icon.ico from app_icon.png when needed.
REM    - Added optional EXE icon support via assets\app_icon.ico in the PyInstaller spec.
REM    - Renamed overlay controls to Preview forward/inverse error.
REM    - Coupled overlay preview-error auto-selection to Output DXF Inverse/Forward mode changes.
REM    - Added preview affine error target scope: None / Overlay only / All layers (except overlay).
REM    - Applied swapped preview affine mapping only to Overlay scope; main layers use normal mapping.
REM    - Coordinates in preview now follow the displayed transformed graph consistently across layers.
REM    - Fixed overlay DXF coordinate preview rendering in the preview dialog.
REM    - Added armed Process/Inverse <-> Process/Forward split-button behavior.
REM
REM  Optional environment variables:
REM    SKIP_INSTALL=1     Skip pip install (use after deps already installed)
REM    PIP_UPGRADE=1      Pass --upgrade when installing requirements.txt
REM    OPEN_DIST=1        Open dist folder after successful build
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
if not exist "assets\app_icon.ico" if not exist "assets\app_icon.png" (
  echo [ERROR] App icon missing. Add assets\app_icon.ico or assets\app_icon.png
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
echo [INFO] Validating runtime imports...
"%VENV_PY%" -c "from affine_core import TpsWarpModel, build_tps_warp_from_affine_residuals; import gui_app"
if errorlevel 1 (
  echo [ERROR] Runtime import check failed.
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

REM --- Icon prep (EXE + runtime) ---
if not exist "assets\app_icon.ico" (
  if exist "assets\app_icon.png" (
    echo [INFO] Generating assets\app_icon.ico from assets\app_icon.png ...
    "%VENV_PY%" -c "from PySide6.QtGui import QImage; import sys; img=QImage(r'assets/app_icon.png'); ok=(not img.isNull()) and img.save(r'assets/app_icon.ico'); sys.exit(0 if ok else 1)"
    if errorlevel 1 (
      echo [ERROR] Failed to generate assets\app_icon.ico from PNG.
      exit /b 1
    )
  )
)
if exist "assets\app_icon.ico" (
  echo [INFO] EXE icon: assets\app_icon.ico
) else (
  echo [ERROR] Icon preparation failed; assets\app_icon.ico not found.
  exit /b 1
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
  if /i "%OPEN_DIST%"=="1" (
    echo [INFO] Opening dist folder...
    start "" "%cd%\%DIST_DIR%"
  )
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
