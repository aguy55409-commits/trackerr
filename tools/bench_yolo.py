"""Benchmark the production DXGI + inference path."""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exoai_tracker.dxgi_capture import Capture
from exoai_tracker.infer import Infer


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def bench(path: Path, capture_size: int, net_size: int, frames: int) -> None:
    print(f"\n=== {path.name} capture={capture_size} net={net_size} ===")
    capture = Capture(size=capture_size)
    capture.start()
    infer = Infer(path, conf=0.30, iou=0.30, imgsz=net_size, warmup=8)
    times: list[float] = []
    ages: list[float] = []
    seqs: set[int] = set()
    det_count = 0
    last_seq = -1
    for i in range(frames + 20):
        frame = capture.latest()
        while frame is None or frame.seq == last_seq:
            time.sleep(0.0002)
            frame = capture.latest()
        if frame is None:
            continue
        last_seq = frame.seq
        t0 = time.perf_counter()
        detections = infer.predict(frame.rgb, frame.left, frame.top)
        ms = (time.perf_counter() - t0) * 1000
        if i >= 20:
            times.append(ms)
            seqs.add(frame.seq)
            det_count += len(detections)
            ages.append((time.perf_counter() - frame.captured_at) * 1000)
    capture_hz = capture.publish_hz
    capture.stop()
    if times:
        print(
            f"  backend={infer.backend} capture={capture_hz:.1f} Hz "
            f"mean={statistics.mean(times):.2f} ms "
            f"p50={statistics.median(times):.2f} "
            f"p95={percentile(times, .95):.2f} "
            f"p99={percentile(times, .99):.2f} "
            f"max={max(times):.2f} ms "
            f"age_p95={percentile(ages, .95):.2f} ms "
            f"unique_seq={len(seqs)}/{len(times)} dets={det_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/al/best.pt")
    parser.add_argument("--capture", type=int, default=448)
    parser.add_argument("--sizes", type=int, nargs="+", default=[320, 448, 640])
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    for size in args.sizes:
        bench(args.model, args.capture, size, args.frames)


if __name__ == "__main__":
    main()
