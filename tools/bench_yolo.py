"""Quick YOLO backend benchmark for available weights."""
from __future__ import annotations

import time
from pathlib import Path

from exoai_tracker.detector import YoloDetector

ROOT = Path(__file__).resolve().parents[1]


def bench(path: Path, fov: int = 140) -> None:
    print(f"\n=== {path.name} ({path.stat().st_size // 1024 // 1024} MB) ===")
    det = YoloDetector(str(path), 0.35, 320)
    times: list[float] = []
    for i in range(20):
        t0 = time.perf_counter()
        dets, _, _ = det.detect(fov, wait_for_new=False)
        ms = (time.perf_counter() - t0) * 1000
        if i >= 5:
            times.append(ms)
            print(f"  {len(dets)} dets {ms:.1f} ms")
    if times:
        avg = sum(times) / len(times)
        print(f"  avg {avg:.1f} ms (~{1000/avg:.0f} Hz)")
    det.close()


def main() -> None:
    for folder in ("models/al", "models/fn"):
        d = ROOT / folder
        for name in ("best.engine", "best.pt"):
            p = d / name
            if p.exists():
                bench(p)


if __name__ == "__main__":
    main()
