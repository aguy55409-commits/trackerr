"""
Train YOLO models for FN, AL, VAL, or R6.

Datasets:
  FN: https://platform.ultralytics.com/tommy-de-sousa/datasets/exoai-dataset-v25-fortnite
  AL: https://platform.ultralytics.com/tommy-de-sousa/datasets/apex-merged-ultralytics-4class
  VAL: https://platform.ultralytics.com/daycien/datasets/valorantv1iyolo26-2
  R6: https://platform.ultralytics.com/jamess/datasets/r6-datasetv8iyolov8

Usage:
  python train.py --preset fn
  python train.py --preset al
  python train.py --preset val
  python train.py --preset r6
  python train.py --preset fn --data path/to/data.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from exoai_tracker.models import MODEL_PRESETS

STATUS_PATH = Path(__file__).resolve().parent / "training_status.json"


def write_status(preset: str, state: str, epochs: int, message: str = "") -> None:
    STATUS_PATH.write_text(
        json.dumps(
            {
                "preset": preset.upper(),
                "state": state,
                "epochs": epochs,
                "message": message,
                "updated_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FN or AL YOLO models")
    parser.add_argument("--preset", choices=["fn", "al", "val", "r6"], default="fn", help="FN, AL, VAL, or R6 model preset")
    parser.add_argument("--data", default="", help="Path to data.yaml")
    parser.add_argument("--model", default="yolo26s.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="CUDA device or cpu")
    parser.add_argument(
        "--enemy-class",
        type=int,
        default=None,
        help="Dataset class ID for enemy (defaults: FN=0, AL=1)",
    )
    args = parser.parse_args()

    preset_key = args.preset.upper()
    preset = MODEL_PRESETS[preset_key]  # type: ignore[index]
    models_dir = Path(__file__).resolve().parent / "models" / args.preset
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        "fn": ("exoai-dataset-v25-fortnite", "tommy-de-sousa", 0),
        "al": ("apex-merged-ultralytics-4class", "tommy-de-sousa", 1),
        "val": ("valorantv1iyolo26-2", "daycien", None),
        "r6": ("r6-datasetv8iyolov8", "jamess", None),
    }[args.preset]
    dataset_slug, dataset_owner, default_enemy = dataset_info
    local_yaml = Path(__file__).resolve().parent / "datasets" / dataset_slug / "data.yaml"
    platform_uri = f"ul://{dataset_owner}/datasets/{dataset_slug}"
    data = args.data or (str(local_yaml) if local_yaml.exists() else platform_uri)
    enemy_class = args.enemy_class if args.enemy_class is not None else default_enemy

    write_status(args.preset, "starting", args.epochs)
    model = YOLO(args.model)
    try:
        write_status(args.preset, "training", args.epochs)
        train_kwargs = dict(
            data=data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            close_mosaic=15,
            patience=30,
            cos_lr=True,
            workers=12,
            project=str(models_dir.parent),
            name=args.preset,
            exist_ok=True,
        )
        if enemy_class is not None:
            train_kwargs["classes"] = [enemy_class]
        results = model.train(**train_kwargs)
    except Exception as exc:
        write_status(args.preset, "failed", args.epochs, str(exc))
        raise

    best = Path(results.save_dir) / "weights" / "best.pt"
    target = models_dir / "best.pt"
    if best.exists():
        target.write_bytes(best.read_bytes())
        print(f"Copied {preset.label} weights to {target}")
    write_status(args.preset, "complete", args.epochs)


if __name__ == "__main__":
    main()
