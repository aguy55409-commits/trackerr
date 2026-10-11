"""Build short- and long-range specialist datasets from the Apex corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = ["teammate", "enemy", "knocked", "other"]
ENEMY_CLASS = 1


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue
        if 0 <= cls < len(CLASS_NAMES) and w > 0 and h > 0:
            rows.append((cls, x, y, w, h))
    return rows


def label_path(root: Path, split: str, image: Path) -> Path:
    return root / "labels" / split / f"{image.stem}.txt"


def records(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / "images" / split
    out: list[tuple[Path, Path]] = []
    for image in image_dir.iterdir():
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        labels = label_path(root, split, image)
        if any(row[0] == ENEMY_CLASS for row in read_labels(labels)):
            out.append((image, labels))
    return sorted(out)


def to_xyxy(
    row: tuple[int, float, float, float, float], width: int, height: int
) -> tuple[int, float, float, float, float]:
    cls, x, y, w, h = row
    return (
        cls,
        (x - w / 2) * width,
        (y - h / 2) * height,
        (x + w / 2) * width,
        (y + h / 2) * height,
    )


def short_transform(
    image: np.ndarray,
    rows: list[tuple[int, float, float, float, float]],
    rng: random.Random,
    size: int,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    height, width = image.shape[:2]
    enemies = [row for row in rows if row[0] == ENEMY_CLASS]
    target = enemies[rng.randrange(len(enemies))]
    _, x1, y1, x2, y2 = to_xyxy(target, width, height)
    target_h = max(2.0, y2 - y1)
    desired_h = rng.uniform(0.30, 0.58)
    crop_size = int(max(target_h / desired_h, min(width, height) * 0.18))
    crop_size = max(32, min(crop_size, width, height))
    cx = (x1 + x2) / 2 + rng.uniform(-0.08, 0.08) * crop_size
    cy = (y1 + y2) / 2 + rng.uniform(-0.08, 0.08) * crop_size
    left = int(max(0, min(width - crop_size, cx - crop_size / 2)))
    top = int(max(0, min(height - crop_size, cy - crop_size / 2)))
    right, bottom = left + crop_size, top + crop_size

    transformed: list[tuple[int, float, float, float, float]] = []
    for row in rows:
        cls, bx1, by1, bx2, by2 = to_xyxy(row, width, height)
        original_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        nx1, ny1 = max(bx1, left), max(by1, top)
        nx2, ny2 = min(bx2, right), min(by2, bottom)
        kept_area = max(0.0, nx2 - nx1) * max(0.0, ny2 - ny1)
        if original_area <= 0 or kept_area / original_area < 0.35:
            continue
        nx1, nx2 = nx1 - left, nx2 - left
        ny1, ny2 = ny1 - top, ny2 - top
        transformed.append(
            (
                cls,
                ((nx1 + nx2) / 2) / crop_size,
                ((ny1 + ny2) / 2) / crop_size,
                (nx2 - nx1) / crop_size,
                (ny2 - ny1) / crop_size,
            )
        )

    crop = image[top:bottom, left:right]
    output = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return output, transformed


def letterbox(
    image: np.ndarray,
    rows: list[tuple[int, float, float, float, float]],
    size: int,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    transformed = [
        (
            cls,
            (left + x * new_w) / size,
            (top + y * new_h) / size,
            w * new_w / size,
            h * new_h / size,
        )
        for cls, x, y, w, h in rows
    ]
    return canvas, transformed


def long_transform(
    image: np.ndarray,
    rows: list[tuple[int, float, float, float, float]],
    rng: random.Random,
    size: int,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    base, rows = letterbox(image, rows, size)
    enemies = [row for row in rows if row[0] == ENEMY_CLASS]
    target = enemies[rng.randrange(len(enemies))]
    desired_h = rng.uniform(0.025, 0.12)
    scale = max(0.28, min(0.88, desired_h / max(target[4], 1e-6)))
    small_size = max(32, round(size * scale))
    small = cv2.resize(base, (small_size, small_size), interpolation=cv2.INTER_AREA)

    mean = tuple(int(v) for v in base.reshape(-1, 3).mean(axis=0))
    canvas = np.full((size, size, 3), mean, dtype=np.uint8)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(
        0, 3.0, canvas.shape
    )
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    left = rng.randrange(0, size - small_size + 1)
    top = rng.randrange(0, size - small_size + 1)
    canvas[top : top + small_size, left : left + small_size] = small
    transformed = [
        (
            cls,
            (left + x * small_size) / size,
            (top + y * small_size) / size,
            w * small_size / size,
            h * small_size / size,
        )
        for cls, x, y, w, h in rows
    ]
    return canvas, transformed


def color_augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.88, 1.12)
    beta = rng.uniform(-10.0, 10.0)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def write_sample(
    source_root: Path,
    destination: Path,
    split: str,
    mode: str,
    record: tuple[Path, Path],
    variant: int,
    size: int,
) -> tuple[bool, list[float]]:
    image_path, labels_path = record
    key = f"{image_path.relative_to(source_root)}:{mode}:{variant}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    output_image = destination / "images" / split / f"{digest}.jpg"
    output_label = destination / "labels" / split / f"{digest}.txt"
    if output_image.exists() and output_label.exists():
        rows = read_labels(output_label)
        return True, [r[4] for r in rows if r[0] == ENEMY_CLASS]

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    rows = read_labels(labels_path)
    if image is None or not any(row[0] == ENEMY_CLASS for row in rows):
        return False, []
    rng = random.Random(int(digest, 16))
    if mode == "short":
        output, transformed = short_transform(image, rows, rng, size)
    else:
        output, transformed = long_transform(image, rows, rng, size)
    enemies = [row for row in transformed if row[0] == ENEMY_CLASS]
    if not enemies:
        return False, []
    output = color_augment(output, rng)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_image), output, [cv2.IMWRITE_JPEG_QUALITY, 90]
    ):
        return False, []
    output_label.write_text(
        "\n".join(
            f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for cls, x, y, w, h in transformed
        )
        + "\n",
        encoding="utf-8",
    )
    return True, [row[4] for row in enemies]


def build_mode(
    source: Path,
    output: Path,
    mode: str,
    counts: dict[str, int],
    size: int,
    workers: int,
) -> dict:
    summary: dict[str, dict] = {}
    for split, target_count in counts.items():
        candidates = records(source, split)
        if not candidates:
            raise RuntimeError(f"No enemy-labeled records in {split}")
        rng = random.Random(4107 + len(split) + (0 if mode == "short" else 1000))
        rng.shuffle(candidates)
        jobs = [
            (candidates[index % len(candidates)], index // len(candidates))
            for index in range(target_count)
        ]
        heights: list[float] = []
        written = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    write_sample,
                    source,
                    output,
                    split,
                    mode,
                    record,
                    variant,
                    size,
                )
                for record, variant in jobs
            ]
            for index, future in enumerate(as_completed(futures), 1):
                ok, enemy_heights = future.result()
                written += int(ok)
                heights.extend(enemy_heights)
                if index % 1000 == 0:
                    print(f"{mode} {split}: {index}/{target_count}", flush=True)
        if written != target_count:
            raise RuntimeError(
                f"{mode}/{split}: wrote {written}, expected {target_count}"
            )
        heights.sort()
        summary[split] = {
            "images": written,
            "enemy_boxes": len(heights),
            "enemy_height_p50": heights[len(heights) // 2] if heights else 0,
            "enemy_height_p90": heights[math.floor(len(heights) * 0.9)]
            if heights
            else 0,
        }

    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "mode": mode,
        "source": str(source.resolve()),
        "source_url": (
            "https://platform.ultralytics.com/tommy-de-sousa/datasets/"
            "apex-merged-ultralytics-4class"
        ),
        "image_size": size,
        "split_policy": "Derived strictly within the source train/val/test split.",
        "transforms": (
            "target-centered crops" if mode == "short" else "downscale and pad"
        ),
        "summary": summary,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--val", type=int, default=2000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(2, os.cpu_count() // 2))
    args = parser.parse_args()
    counts = {"train": args.train, "val": args.val, "test": args.test}
    for mode in ("short", "long"):
        output = args.output_root / f"apex-{mode}-range-23k"
        manifest = build_mode(
            args.source, output, mode, counts, args.size, args.workers
        )
        print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
