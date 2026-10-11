"""Download a large file with parallel HTTP range requests."""
from __future__ import annotations

import argparse
import math
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def content_length(url: str) -> int:
    request = urllib.request.Request(
        url,
        headers={
            "Range": "bytes=0-0",
            "User-Agent": "pip/24.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        return int(response.headers["Content-Length"])


def fetch_part(url: str, path: Path, start: int, end: int) -> None:
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "pip/24.0",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"Server ignored range request: {response.status}")
        with path.open("wb") as output:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                output.write(chunk)
    if path.stat().st_size != expected:
        raise RuntimeError(f"{path.name}: incomplete range")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--connections", type=int, default=16)
    args = parser.parse_args()

    total = content_length(args.url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part_size = math.ceil(total / args.connections)
    parts: list[tuple[Path, int, int]] = []
    for index in range(args.connections):
        start = index * part_size
        end = min(total - 1, start + part_size - 1)
        if start > end:
            break
        parts.append(
            (args.output.with_suffix(args.output.suffix + f".part{index:02d}"), start, end)
        )

    finished = threading.Event()

    def progress() -> None:
        while not finished.wait(5):
            downloaded = sum(p.stat().st_size for p, _, _ in parts if p.exists())
            print(
                f"{downloaded / total * 100:5.1f}% "
                f"({downloaded / (1 << 20):.0f}/{total / (1 << 20):.0f} MiB)",
                flush=True,
            )

    reporter = threading.Thread(target=progress, daemon=True)
    reporter.start()
    try:
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            futures = [
                pool.submit(fetch_part, args.url, path, start, end)
                for path, start, end in parts
            ]
            for future in futures:
                future.result()
    finally:
        finished.set()
        reporter.join(timeout=1)

    with args.output.open("wb") as output:
        for path, _, _ in parts:
            with path.open("rb") as source:
                while True:
                    chunk = source.read(4 << 20)
                    if not chunk:
                        break
                    output.write(chunk)
            path.unlink()
    if args.output.stat().st_size != total:
        raise RuntimeError("Combined file has the wrong size")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
