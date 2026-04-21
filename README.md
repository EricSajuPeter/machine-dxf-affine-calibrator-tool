# Affine Calibration GUI

Desktop application for:
- 2D affine calibration from center-first input plus variable matched pairs (minimum 3)
- Measured point rectification (`measured -> ideal`)
- DXF reverse-distortion compensation (`ideal -> pre-distorted DXF`)

## Requirements

- Windows
- Python 3.10+ recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run GUI

```bash
python gui_app.py
```

## Workflow

1. Enter calibration points:
   - First row is center:
     - Ideal center is fixed at `0,0`
     - Measured center is optional
   - Add matched ideal/measured pairs (minimum 3, unlimited additional rows)
   - Pair order must match physical correspondences
2. Click **Solve Calibration**.
3. The **Calibration shape** plot (measured space) overlays as-measured path (red), model prediction (blue), and compensated shape (green) after solve. Requires `matplotlib`.
4. The app shows predicted measured center from solved transform (and center delta if measured center was provided).
5. Rectify a measured point in **Point Rectification**.
6. For full design compensation:
   - Select input ideal DXF
   - Select output DXF path
   - Click **Process DXF**

The output DXF is pre-distorted so fabrication machine distortion is cancelled at output.

If the shape plot does not appear, install matplotlib: `pip install matplotlib`.

## Build Windows EXE

From this folder in **Command Prompt** or PowerShell:

```bat
build_exe.bat
```

This creates/uses `.venv`, installs `requirements.txt`, and runs PyInstaller with `AffineCalibratorGUI.spec` (single-file, no console window).

Output:

- `dist\AffineCalibratorGUI.exe`

**Manual build** (if you prefer not to use the batch file):

```bat
.venv\Scripts\activate
pip install -r requirements.txt
python -m PyInstaller --noconfirm --clean AffineCalibratorGUI.spec
```

First build can take several minutes; the EXE is typically tens to hundreds of MB (bundles Python, PySide6, matplotlib, numpy, ezdxf).

## Web App (GitHub Pages)

This repository now also includes a modern, scroll-first browser app in `web/`.

### Run locally

```bash
cd web
npm install
npm run dev
```

### Build

```bash
cd web
npm install
npm run build
```

### Deploy

A GitHub Actions workflow at `.github/workflows/web-pages.yml` builds and deploys the `web` app to GitHub Pages on pushes to `main`.

### Web app scope (current)

- In-browser affine solve from matched point pairs
- DXF upload/parsing (LINE/LWPOLYLINE/POLYLINE focus)
- Layered comparison view (Input, Output, Distorted)
- Compensated DXF download

Processing is client-side; uploaded DXF stays on the user's machine.

## Notes

- DXF processing relies on `ezdxf`.
- Unsupported complex entities are converted to polyline approximations when needed.
- Review warnings in the GUI log after DXF processing.
