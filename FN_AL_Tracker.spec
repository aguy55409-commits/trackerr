# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

import webview

block_cipher = None

hiddenimports = [
    "exoai_tracker",
    "exoai_tracker.binds",
    "exoai_tracker.config",
    "exoai_tracker.detector",
    "exoai_tracker.gui",
    "exoai_tracker.main",
    "exoai_tracker.models",
    "exoai_tracker.mouse",
    "exoai_tracker.overlay",
    "exoai_tracker.paths",
    "exoai_tracker.refine",
    "exoai_tracker.scan",
    "exoai_tracker.tracker",
    "exoai_tracker.rzctl",
    "exoai_tracker.rzctl.rzctl",
    "exoai_tracker.rzctl.rzctl_nt",
    "webview",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "clr",
    "clr_loader",
    "pythonnet",
    "bottle",
    "proxy_tools",
    "ultralytics",
    "ultralytics.nn",
    "ultralytics.utils",
    "ultralytics.models",
    "ultralytics.engine",
    "ultralytics.cfg",
    "torch",
    "torchvision",
    "cv2",
    "PIL",
    "mss",
    "dxcam",
    "keyboard",
    "numpy",
    "yaml",
    "scipy",
    "matplotlib",
    "pandas",
    "win32api",
    "win32con",
    "win32gui",
    "pywintypes",
]

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=[],
    datas=[("exoai_tracker/ui", "exoai_tracker/ui")],
    hiddenimports=hiddenimports,
    hookspath=[str(Path(webview.__file__).resolve().parent / "__pyinstaller")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FN AL Tracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    uac_admin=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FN AL Tracker",
)
