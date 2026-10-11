"""Export a fixed-size ONNX profile without replacing the default fallback."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--size", type=int, required=True)
    args = parser.parse_args()

    default_onnx = args.weights.with_suffix(".onnx")
    backup = default_onnx.with_name(default_onnx.stem + "_export_backup.onnx")
    destination = default_onnx.with_name(f"{default_onnx.stem}_{args.size}.onnx")
    if default_onnx.exists():
        default_onnx.replace(backup)
    try:
        exported = Path(
            YOLO(str(args.weights)).export(
                format="onnx",
                imgsz=args.size,
                half=False,
                dynamic=False,
                simplify=True,
                device="cpu",
            )
        )
        exported.replace(destination)
        print(destination)
    finally:
        if backup.exists():
            backup.replace(default_onnx)


if __name__ == "__main__":
    main()
