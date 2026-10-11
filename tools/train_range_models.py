"""Train short, long, then combined Apex detectors to a target mAP50."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "training_range_status.json"


def status(**values) -> None:
    values["updated_at"] = time.time()
    STATUS.write_text(json.dumps(values, indent=2), encoding="utf-8")


def write_combined_yaml(
    source: Path, short: Path, long: Path, output: Path
) -> Path:
    data = {
        "train": [
            str((source / "images/train").resolve()),
            str((short / "images/train").resolve()),
            str((long / "images/train").resolve()),
        ],
        "val": [
            str((source / "images/val").resolve()),
            str((short / "images/val").resolve()),
            str((long / "images/val").resolve()),
        ],
        "test": [
            str((source / "images/test").resolve()),
            str((short / "images/test").resolve()),
            str((long / "images/test").resolve()),
        ],
        "names": {
            0: "teammate",
            1: "enemy",
            2: "knocked",
            3: "other",
        },
    }
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def map50(model_path: Path, data: Path, batch: int, workers: int) -> float:
    metrics = YOLO(str(model_path)).val(
        data=str(data),
        split="test",
        imgsz=640,
        batch=batch,
        device=0,
        workers=workers,
        classes=[1],
        plots=False,
        verbose=False,
    )
    return float(metrics.box.map50)


def train_until(
    name: str,
    data: Path,
    initial: Path,
    destination: Path,
    target: float,
    round_epochs: int,
    max_epochs: int,
    batch: int,
    workers: int,
) -> tuple[Path, float, int]:
    current = initial
    best_score = 0.0
    epochs_used = 0
    round_index = 0
    if destination.exists():
        best_score = map50(destination, data, batch, workers)
        current = destination
        status(
            stage="checkpoint_validated",
            model=name,
            epochs_used=0,
            target_map50=target,
            best_map50=best_score,
        )
        if best_score >= target:
            return destination, best_score, epochs_used
    while epochs_used < max_epochs and best_score < target:
        round_index += 1
        epochs = min(round_epochs, max_epochs - epochs_used)
        status(
            stage="training",
            model=name,
            round=round_index,
            epochs_used=epochs_used,
            target_map50=target,
            best_map50=best_score,
        )
        run = YOLO(str(current)).train(
            data=str(data),
            epochs=epochs,
            imgsz=640,
            batch=batch,
            device=0,
            workers=workers,
            classes=[1],
            close_mosaic=10,
            patience=max(15, epochs // 3),
            cos_lr=True,
            amp=True,
            cache="disk",
            project=str(ROOT / "models" / "range_training"),
            name=f"{name}_round{round_index}",
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        epochs_used += epochs
        candidate = Path(run.save_dir) / "weights" / "best.pt"
        if not candidate.exists():
            raise RuntimeError(f"{name}: training did not produce best.pt")
        score = map50(candidate, data, batch, workers)
        if score >= best_score:
            best_score = score
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            current = destination
        else:
            current = destination
        status(
            stage="validated",
            model=name,
            round=round_index,
            epochs_used=epochs_used,
            target_map50=target,
            best_map50=best_score,
        )
    return destination, best_score, epochs_used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--range-root", type=Path, required=True)
    parser.add_argument("--initial", type=Path, default=ROOT / "models/al/best.pt")
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--round-epochs", type=int, default=60)
    parser.add_argument("--max-epochs", type=int, default=240)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    short_root = args.range_root / "apex-short-range-23k"
    long_root = args.range_root / "apex-long-range-23k"
    short_yaml = short_root / "data.yaml"
    long_yaml = long_root / "data.yaml"
    combined_yaml = write_combined_yaml(
        args.source, short_root, long_root, args.range_root / "apex-combined.yaml"
    )

    outputs: dict[str, dict] = {}
    for name, data in (("short", short_yaml), ("long", long_yaml)):
        model, score, epochs = train_until(
            name,
            data,
            args.initial,
            ROOT / f"models/al/best_{name}.pt",
            args.target,
            args.round_epochs,
            args.max_epochs,
            args.batch,
            args.workers,
        )
        outputs[name] = {
            "model": str(model),
            "map50": score,
            "epochs": epochs,
        }

    model, score, epochs = train_until(
        "combined",
        combined_yaml,
        args.initial,
        ROOT / "models/al/best_combined.pt",
        args.target,
        args.round_epochs,
        args.max_epochs,
        args.batch,
        args.workers,
    )
    outputs["combined"] = {
        "model": str(model),
        "map50": score,
        "epochs": epochs,
    }
    status(stage="complete", target_map50=args.target, results=outputs)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
