"""Small launcher. Installs Python packages from requirements.txt, then starts the app.

The executable stays in this folder. It does not carry the packages itself.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

PYTHON_VERSION = "3.11.9"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-amd64.exe"
STAMP_NAME = ".deps_ok"


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def title(text: str) -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleTitleW(text)
        except Exception:
            pass


def find_python() -> Path | None:
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(local) / "Programs" / "Python" / "Python311" / "python.exe",
        Path(r"C:\Program Files\Python311\python.exe"),
        Path(r"C:\Python311\python.exe"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    try:
        out = subprocess.check_output(
            ["py", "-3.11", "-c", "import sys; print(sys.executable)"],
            text=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None
    found = Path(out.strip())
    return found if found.is_file() else None


def download(url: str, dest: Path) -> None:
    def report(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(total, block * block_size)
        pct = done * 100 // total
        print(f"\r  {pct:3d}%  {done // (1024 * 1024)} / {total // (1024 * 1024)} MB", end="", flush=True)

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest, report)
    print()


def ensure_python() -> Path:
    found = find_python()
    if found is not None:
        print(f"Python: {found}")
        return found

    print("Python 3.11 is not installed. Downloading it.")
    installer = app_dir() / f"python-{PYTHON_VERSION}-amd64.exe"
    download(PYTHON_URL, installer)
    target = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311"
    print("Installing Python 3.11. This can take a minute.")
    completed = subprocess.run(
        [
            str(installer),
            "/quiet",
            "InstallAllUsers=0",
            "PrependPath=1",
            "Include_pip=1",
            "Include_test=0",
            "Include_launcher=1",
            f"TargetDir={target}",
        ]
    )
    if completed.returncode != 0:
        raise SystemExit(f"Python installer failed ({completed.returncode}).")
    found = find_python()
    if found is None:
        raise SystemExit("Python installed, but python.exe was not found.")
    print(f"Python: {found}")
    return found


def requirements_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_dependencies(python: Path, root: Path) -> None:
    requirements = root / "requirements.txt"
    if not requirements.is_file():
        raise SystemExit(f"Missing {requirements}")
    stamp = root / STAMP_NAME
    digest = requirements_hash(requirements)
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == digest:
        print("Dependencies already installed.")
        return

    print("Installing dependencies. The first run downloads several GB and can take a while.")
    pip = [str(python), "-m", "pip", "install", "--upgrade", "pip"]
    if subprocess.run(pip).returncode != 0:
        raise SystemExit("Could not update pip.")
    install = [str(python), "-m", "pip", "install", "-r", str(requirements)]
    if subprocess.run(install, cwd=root).returncode != 0:
        raise SystemExit("Dependency install failed. See the messages above.")
    stamp.write_text(digest, encoding="utf-8")
    print("Dependencies installed.")


def launch(python: Path, root: Path) -> None:
    script = root / "run.py"
    if not script.is_file():
        raise SystemExit(f"Missing {script}")
    print("Starting Neverwin.")
    subprocess.Popen([str(python), str(script)], cwd=root)


def main() -> None:
    title("Neverwin")
    root = app_dir()
    os.chdir(root)
    print(f"Folder: {root}")
    try:
        python = ensure_python()
        install_dependencies(python, root)
        launch(python, root)
    except SystemExit as exc:
        print(exc)
        input("Press Enter to close.")
        raise
    except Exception as exc:
        print(f"Failed: {exc}")
        input("Press Enter to close.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
