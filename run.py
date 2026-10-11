"""Launcher — clean dual-thread TensorRT tracker."""
import ctypes
import logging
import os
import sys
import threading
from pathlib import Path

# Hide console window on Windows
if sys.platform == "win32":
    try:
        ctypes.windll.user32.ShowWindow(
            ctypes.windll.kernel32.GetConsoleWindow(), 0
        )
    except Exception:
        pass

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "0")
os.environ.setdefault("YOLO_VERBOSE", "False")

logging.basicConfig(
    filename=Path(__file__).resolve().parent / "neverwin.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
)


def _log_exception(exc_type, exc_value, exc_traceback) -> None:
    logging.getLogger("neverwin").critical(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


sys.excepthook = _log_exception


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    _log_exception(args.exc_type, args.exc_value, args.exc_traceback)


threading.excepthook = _log_thread_exception


def _warm_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        pass


threading.Thread(target=_warm_torch, name="torch-warm", daemon=True).start()

from exoai_tracker.main import main

if __name__ == "__main__":
    main()
