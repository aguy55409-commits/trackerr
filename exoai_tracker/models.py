from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from exoai_tracker.paths import app_dir, resolve_data_path

ModelId = Literal["AL"]

MODEL_IDS: tuple[ModelId, ...] = ("AL",)
YOLO26_BASE = "yolo26s.pt"
REGISTRY_PATH = app_dir() / "models" / "registry.json"


@dataclass(frozen=True)
class ModelPreset:
    id: ModelId
    label: str
    model_path: str
    dataset_url: str
    description: str
    target_class_ids: tuple[int, ...] | None = None
    capture_size: int = 320


MODEL_PRESETS: dict[ModelId, ModelPreset] = {
    "AL": ModelPreset(
        id="AL",
        label="Apex Legends",
        model_path="models/al/best.pt",
        dataset_url="https://platform.ultralytics.com/tommy-de-sousa/datasets/apex-merged-ultralytics-4class",
        description="YOLO · Apex 4-class detection",
        target_class_ids=None,
    ),
}


@lru_cache(maxsize=1)
def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_registry_entry(model_id: str) -> dict | None:
    return load_registry().get(normalize_model_id(model_id))


def apply_registry_presets() -> None:
    """Merge models/registry.json into in-memory presets."""
    registry = load_registry()
    for model_id, entry in registry.items():
        if model_id not in MODEL_PRESETS:
            continue
        old = MODEL_PRESETS[model_id]  # type: ignore[index]
        class_ids = entry.get("target_class_ids")
        if class_ids is not None:
            class_ids = tuple(class_ids)
        weights = entry.get("weights") or []
        MODEL_PRESETS[model_id] = ModelPreset(  # type: ignore[index]
            id=old.id,
            label=entry.get("label", old.label),
            model_path=weights[0] if weights else old.model_path,
            dataset_url=old.dataset_url,
            description=entry.get("description", old.description),
            target_class_ids=class_ids if class_ids is not None else old.target_class_ids,
            capture_size=int(entry.get("capture_size", old.capture_size)),
        )


def normalize_model_id(value: str) -> ModelId:
    return "AL"


def is_color_mode(model_id: str) -> bool:
    return False


def get_preset(model_id: str) -> ModelPreset:
    return MODEL_PRESETS[normalize_model_id(model_id)]


def get_capture_size(model_id: str) -> int:
    entry = get_registry_entry(model_id)
    if entry and "capture_size" in entry:
        return max(320, min(1280, int(entry["capture_size"])))
    return get_preset(model_id).capture_size


def resolve_fastest_weights(model_dir: Path, configured: Path | None = None) -> Path:
    """Prefer a local TensorRT engine, then the configured/PT fallback."""
    folder = model_dir if model_dir.is_dir() else model_dir.parent

    if (
        configured is not None
        and configured.exists()
        and configured.stem.startswith("best_combined")
    ):
        return configured

    # FP16 engine first (half the inference time), then FP32, then generic.
    for name in ("best_fp16.engine", "best_fp32.engine", "best.engine", "fast.engine"):
        p = folder / name
        if p.exists():
            return p

    if configured is not None and configured.exists():
        return configured

    for name in (
        "best.pt",
        "weights/best.pt",
        "fast.pt",
    ):
        p = folder / name
        if p.exists():
            return p

    pt_files = list(folder.glob("*.pt"))
    if pt_files:
        return pt_files[0]
    return folder / "best.pt"


def resolve_preset_weights(active_model: str, configured_path: str = "") -> Path:
    """Resolve trained weights: TensorRT engine preferred."""
    model_id = normalize_model_id(active_model)
    preset = get_preset(model_id)
    configured = resolve_data_path(configured_path or preset.model_path)
    folder = configured.parent if configured.parent.exists() else resolve_data_path(preset.model_path).parent
    return resolve_fastest_weights(folder, configured)


apply_registry_presets()
