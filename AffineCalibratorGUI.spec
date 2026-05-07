# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AffineCalibratorGUI (PySide6 + matplotlib + numpy + ezdxf)."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

# Matplotlib needs font/config data at runtime for embedded plots.
_mpl_datas = collect_data_files("matplotlib")
_project_root = Path(SPECPATH).resolve()
_icon_path = _project_root / "assets" / "app_icon.ico"
_icon_png_path = _project_root / "assets" / "app_icon.png"
if not _icon_path.exists():
    raise FileNotFoundError(
        f"Missing required icon: {_icon_path}. Run build_exe.bat to generate it from assets/app_icon.png if needed."
    )
_exe_icon = str(_icon_path)
_datas = list(_mpl_datas)
if _icon_path.exists():
    _datas.append((str(_icon_path), "assets"))
if _icon_png_path.exists():
    _datas.append((str(_icon_png_path), "assets"))

a = Analysis(
    ["gui_app.py"],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_qt",
        "matplotlib.figure",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AffineCalibratorGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_exe_icon,
)
