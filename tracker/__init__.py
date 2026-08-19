"""Clean TensorRT YOLO aim tracker — minimal core."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

PRESETS = {
    "AL": MODELS / "al" / "best.engine",
    "FN": MODELS / "fn" / "best.engine",
    "VAL": MODELS / "val" / "best.engine",
    "R6": MODELS / "r6" / "best.engine",
}


def resolve_engine(preset: str) -> Path:
    key = preset.upper()
    path = PRESETS.get(key, PRESETS["AL"])
    if path.exists():
        return path
    # fall back to .pt if engine missing
    pt = path.with_suffix(".pt")
    if pt.exists():
        return pt
    raise FileNotFoundError(f"No model for {key}: {path}")
