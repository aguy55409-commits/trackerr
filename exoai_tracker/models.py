from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from exoai_tracker.detector import YoloDetector
from exoai_tracker.paths import resolve_data_path

ModelId = Literal["FN", "AL", "VAL", "R6", "COLOR"]

MODEL_IDS: tuple[ModelId, ...] = ("FN", "AL", "VAL", "R6", "COLOR")


@dataclass(frozen=True)
class ModelPreset:
    id: ModelId
    label: str
    model_path: str
    dataset_url: str
    description: str
    target_class_ids: tuple[int, ...] | None = None


MODEL_PRESETS: dict[ModelId, ModelPreset] = {
    "FN": ModelPreset(
        id="FN",
        label="FN",
        model_path="models/fn/best.engine",
        dataset_url="https://platform.ultralytics.com/tommy-de-sousa/datasets/exoai-dataset-v25-fortnite",
        description="Fortnite · Body detection",
        target_class_ids=None,
    ),
    "AL": ModelPreset(
        id="AL",
        label="AL",
        model_path="models/al/best.engine",
        dataset_url="https://platform.ultralytics.com/tommy-de-sousa/datasets/apex-merged-ultralytics-4class",
        description="Apex Merged · 4-class (enemy, teammate, knocked, etc.)",
        target_class_ids=None,
    ),
    "VAL": ModelPreset(
        id="VAL",
        label="VAL",
        model_path="models/val/best.engine",
        dataset_url="https://platform.ultralytics.com/daycien/datasets/valorantv1iyolo26-2",
        description="Valorant v1 · Body & Head detection",
        target_class_ids=None,  # 0=body, 1=head - detect both
    ),
    "R6": ModelPreset(
        id="R6",
        label="R6",
        model_path="models/r6/best.engine",
        dataset_url="https://platform.ultralytics.com/jamess/datasets/r6-datasetv8iyolov8",
        description="R6 Siege · Ally / enemy / head (skip allies, aim heads)",
        target_class_ids=None,
    ),
    "COLOR": ModelPreset(
        id="COLOR",
        label="COLOR",
        model_path="",  # No model needed
        dataset_url="",
        description="Color detection · Ultra fast · No AI needed",
        target_class_ids=None,
    ),
}


def normalize_model_id(value: str) -> ModelId:
    upper = value.upper()
    if upper in MODEL_PRESETS:
        return upper  # type: ignore[return-value]
    return "FN"


def is_color_mode(model_id: str) -> bool:
    return normalize_model_id(model_id) == "COLOR"


def get_preset(model_id: str) -> ModelPreset:
    return MODEL_PRESETS[normalize_model_id(model_id)]


def resolve_fastest_weights(model_dir: Path, configured: Path | None = None) -> Path:
    """
    Pick the fastest Ultralytics YOLO backend for a model folder.
    Order: TensorRT engine -> trained .pt -> weights/best.pt.
    """
    folder = model_dir if model_dir.is_dir() else model_dir.parent
    engine = YoloDetector.resolve_fastest_engine(folder)
    if engine is not None:
        return Path(engine)

    candidates: list[Path] = []
    if configured is not None and configured.exists():
        candidates.append(configured)
    for name in ("best.pt", "weights/best.pt"):
        p = folder / name
        if p.exists() and p not in candidates:
            candidates.append(p)
    if candidates:
        return candidates[0]
    if configured is not None:
        return configured
    return folder / "best.pt"


def resolve_preset_weights(active_model: str, configured_path: str) -> Path:
    preset = get_preset(active_model)
    configured = resolve_data_path(configured_path or preset.model_path)
    folder = configured.parent if configured.parent.exists() else resolve_data_path(preset.model_path).parent
    return resolve_fastest_weights(folder, configured)
