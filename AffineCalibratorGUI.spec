# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AffineCalibratorGUI (PySide6 + matplotlib + numpy + ezdxf)."""

from PyInstaller.utils.hooks import collect_data_files

# Matplotlib needs font/config data at runtime for embedded plots.
_mpl_datas = collect_data_files("matplotlib")

a = Analysis(
    ["gui_app.py"],
    pathex=[],
    binaries=[],
    datas=_mpl_datas,
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
)
