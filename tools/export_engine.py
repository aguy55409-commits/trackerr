"""Export TensorRT engines from trained YOLO weights (fastest inference path)."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def export_one(pt_path: Path, imgsz: int = 320, half: bool = True) -> Path:
    print(f"Exporting {pt_path} -> TensorRT engine (imgsz={imgsz}, half={half})")
    model = YOLO(str(pt_path), task="detect")
    out = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        simplify=True,
        workspace=4,
        batch=1,
        dynamic=False,
        verbose=False,
    )
    engine = Path(out)
    target = pt_path.with_name("best.engine")
    if engine.resolve() != target.resolve():
        target.write_bytes(engine.read_bytes())
        print(f"Copied to {target}")
    else:
        print(f"Written {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--preset", choices=("al", "fn", "both"), default="both")
    args = parser.parse_args()

    targets: list[Path] = []
    if args.preset in ("al", "both"):
        targets.append(ROOT / "models" / "al" / "best.pt")
    if args.preset in ("fn", "both"):
        targets.append(ROOT / "models" / "fn" / "best.pt")

    for pt in targets:
        if not pt.exists():
            print(f"Skip missing {pt}")
            continue
        export_one(pt, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
