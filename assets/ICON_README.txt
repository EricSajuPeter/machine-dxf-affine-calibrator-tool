Place your app icon in the assets folder:
  assets/app_icon.ico   (preferred)
or
  assets/app_icon.png   (build script auto-converts to .ico)

Notes:
- Format must be .ico for reliable PyInstaller Windows builds.
- Recommended sizes inside the ICO: 16, 24, 32, 48, 64, 128, 256.
- Runtime icon (window/taskbar) is loaded from app_icon.ico, then app_icon.png as fallback.
