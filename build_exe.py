"""Build FN AL Tracker executable and stage runtime files."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "FN AL Tracker"
SPEC = ROOT / "FN_AL_Tracker.spec"

STAGE_ITEMS = (
    "config.json",
    "models",
)


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", str(SPEC)],
        cwd=ROOT,
        check=True,
    )

    if not DIST.exists():
        raise SystemExit(f"Build output missing: {DIST}")

    for name in STAGE_ITEMS:
        src = ROOT / name
        dst = DIST / name
        if not src.exists():
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    print(f"\nBuilt: {DIST / 'FN AL Tracker.exe'}")
    print("Run as Administrator for global keybind detection.")


if __name__ == "__main__":
    main()
