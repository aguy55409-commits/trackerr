from __future__ import annotations

import csv
import ctypes
import json
import math
import subprocess
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk


ROOT = Path(__file__).resolve().parent
STATUS_PATH = ROOT / "training_status.json"
TOTAL_EPOCHS = 120
DATASETS = {
    "FN": {"path": ROOT / "models" / "fn", "images": 22_710},
    "AL": {"path": ROOT / "models" / "al", "images": 35_108},
}


class FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class CpuMeter:
    def __init__(self) -> None:
        self._last = self._sample()

    @staticmethod
    def _value(value: FileTime) -> int:
        return (value.high << 32) | value.low

    def _sample(self) -> tuple[int, int, int]:
        idle, kernel, user = FileTime(), FileTime(), FileTime()
        ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        return self._value(idle), self._value(kernel), self._value(user)

    def percent(self) -> float:
        current = self._sample()
        idle = current[0] - self._last[0]
        total = (current[1] - self._last[1]) + (current[2] - self._last[2])
        self._last = current
        return max(0.0, min(100.0, (1.0 - idle / total) * 100.0)) if total else 0.0


def gpu_stats() -> tuple[int, int, int, int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=True,
        )
        values = [int(float(value.strip())) for value in result.stdout.splitlines()[0].split(",")]
        return values[0], values[1], values[2], values[3]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def fmt_time(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def load_results(folder: Path) -> list[dict[str, str]]:
    path = folder / "results.csv"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = []
            for row in csv.DictReader(handle):
                rows.append({str(key).strip(): str(value).strip() for key, value in row.items()})
            return rows
    except (OSError, csv.Error):
        return []


def load_status() -> dict[str, object]:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def number(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        try:
            return float(row[name])
        except (KeyError, TypeError, ValueError):
            continue
    return None


class ProgressWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ExoAI Training Progress")
        self.root.geometry("620x455")
        self.root.minsize(560, 430)
        self.root.configure(bg="#101317")
        self.root.attributes("-topmost", True)
        self.cpu = CpuMeter()

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Train.Horizontal.TProgressbar",
            troughcolor="#272d35",
            background="#28c78c",
            bordercolor="#272d35",
            lightcolor="#28c78c",
            darkcolor="#28c78c",
        )

        self.status = tk.StringVar(value="Reading training state...")
        self.model = tk.StringVar(value="--")
        self.epoch = tk.StringVar(value="Epoch -- / 120")
        self.overall_label = tk.StringVar(value="TRAINING OVERALL")
        self.total_text = tk.StringVar(value="0.0% total")
        self.elapsed = tk.StringVar(value="--")
        self.eta = tk.StringVar(value="Calculating...")
        self.metrics = tk.StringVar(value="Metrics appear after epoch 1")
        self.hardware = tk.StringVar(value="Reading GPU and CPU...")
        self.updated = tk.StringVar(value="")

        frame = tk.Frame(self.root, bg="#101317", padx=28, pady=24)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="EXOAI MODEL TRAINING",
            bg="#101317",
            fg="#8e9aa8",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w")

        header = tk.Frame(frame, bg="#101317")
        header.pack(fill="x", pady=(8, 4))
        tk.Label(
            header,
            textvariable=self.model,
            bg="#101317",
            fg="#f4f7fa",
            font=("Segoe UI Semibold", 25),
        ).pack(side="left")
        tk.Label(
            header,
            textvariable=self.status,
            bg="#101317",
            fg="#28c78c",
            font=("Segoe UI", 11),
        ).pack(side="right", pady=(10, 0))

        tk.Label(
            frame,
            textvariable=self.epoch,
            bg="#101317",
            fg="#c7ced7",
            font=("Segoe UI", 12),
        ).pack(anchor="w", pady=(8, 6))
        self.current_bar = ttk.Progressbar(
            frame, style="Train.Horizontal.TProgressbar", maximum=100
        )
        self.current_bar.pack(fill="x", ipady=5)

        total_row = tk.Frame(frame, bg="#101317")
        total_row.pack(fill="x", pady=(24, 6))
        tk.Label(
            total_row,
            textvariable=self.overall_label,
            bg="#101317",
            fg="#8e9aa8",
            font=("Segoe UI Semibold", 10),
        ).pack(side="left")
        tk.Label(
            total_row,
            textvariable=self.total_text,
            bg="#101317",
            fg="#c7ced7",
            font=("Segoe UI", 10),
        ).pack(side="right")
        self.total_bar = ttk.Progressbar(
            frame, style="Train.Horizontal.TProgressbar", maximum=100
        )
        self.total_bar.pack(fill="x", ipady=4)

        times = tk.Frame(frame, bg="#101317")
        times.pack(fill="x", pady=(24, 18))
        self._stat(times, "ELAPSED", self.elapsed).pack(side="left", fill="x", expand=True)
        self._stat(times, "ESTIMATED REMAINING", self.eta).pack(
            side="left", fill="x", expand=True
        )

        tk.Label(
            frame,
            textvariable=self.metrics,
            bg="#1a1f25",
            fg="#c7ced7",
            padx=14,
            pady=12,
            anchor="w",
            font=("Consolas", 10),
        ).pack(fill="x")
        tk.Label(
            frame,
            textvariable=self.hardware,
            bg="#101317",
            fg="#8e9aa8",
            pady=10,
            anchor="w",
            font=("Consolas", 10),
        ).pack(fill="x")
        tk.Label(
            frame,
            textvariable=self.updated,
            bg="#101317",
            fg="#66717d",
            font=("Segoe UI", 9),
        ).pack(anchor="e", pady=(10, 0))

        self.refresh()

    @staticmethod
    def _stat(parent: tk.Widget, title: str, variable: tk.StringVar) -> tk.Frame:
        frame = tk.Frame(parent, bg="#101317")
        tk.Label(
            frame,
            text=title,
            bg="#101317",
            fg="#66717d",
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w")
        tk.Label(
            frame,
            textvariable=variable,
            bg="#101317",
            fg="#f4f7fa",
            font=("Segoe UI Semibold", 16),
        ).pack(anchor="w", pady=(3, 0))
        return frame

    def refresh(self) -> None:
        rows = {name: load_results(info["path"]) for name, info in DATASETS.items()}
        epochs = {name: min(TOTAL_EPOCHS, len(values)) for name, values in rows.items()}
        status = load_status()
        requested = str(status.get("preset", "")).upper()
        if requested in DATASETS:
            active = requested
        elif epochs["AL"] > 0 or epochs["FN"] >= TOTAL_EPOCHS:
            active = "AL"
        else:
            active = "FN"

        configured_epochs = int(status.get("epochs", TOTAL_EPOCHS))
        total_units = DATASETS[active]["images"] * configured_epochs
        done_units = DATASETS[active]["images"] * epochs[active]
        overall = done_units / total_units * 100 if total_units else 0.0

        current_epoch = epochs[active]
        self.model.set(f"{active} · YOLO26s")
        self.epoch.set(f"Epoch {current_epoch} / {configured_epochs}")
        self.current_bar["value"] = current_epoch / configured_epochs * 100
        self.total_bar["value"] = overall
        self.overall_label.set(f"{active} TRAINING OVERALL")
        self.total_text.set(f"{overall:.1f}% total")

        elapsed = 0.0
        for name in ("FN", "AL"):
            if rows[name]:
                elapsed += number(rows[name][-1], "time") or 0.0
        self.elapsed.set(fmt_time(elapsed))

        if done_units > 0 and elapsed > 0:
            rate = done_units / elapsed
            self.eta.set(fmt_time((total_units - done_units) / rate))
        else:
            self.eta.set("After epoch 1")

        state = str(status.get("state", ""))
        if state == "complete":
            self.status.set("Complete")
            self.eta.set("Done")
        elif state == "failed":
            self.status.set("Stopped with error")
        elif current_epoch == 0:
            self.status.set("Preparing first epoch")
        else:
            self.status.set("Training on RTX 4080")

        latest = rows[active][-1] if rows[active] else None
        if latest:
            box = number(latest, "train/box_loss")
            precision = number(latest, "metrics/precision(B)", "metrics/precision")
            recall = number(latest, "metrics/recall(B)", "metrics/recall")
            map50 = number(latest, "metrics/mAP50(B)", "metrics/mAP50")
            parts = []
            if box is not None:
                parts.append(f"box loss {box:.3f}")
            if precision is not None:
                parts.append(f"precision {precision:.1%}")
            if recall is not None:
                parts.append(f"recall {recall:.1%}")
            if map50 is not None:
                parts.append(f"mAP50 {map50:.1%}")
            self.metrics.set("   ·   ".join(parts) or "Epoch metrics recorded")
        else:
            self.metrics.set("Metrics appear after epoch 1")

        gpu = gpu_stats()
        cpu = self.cpu.percent()
        if gpu:
            gpu_use, used, total, temperature = gpu
            self.hardware.set(
                f"GPU {gpu_use:>3}%   VRAM {used / 1024:.1f}/{total / 1024:.1f} GB"
                f"   TEMP {temperature}°C   CPU {cpu:.0f}%"
            )
        else:
            self.hardware.set(f"GPU unavailable   CPU {cpu:.0f}%")

        self.updated.set(f"Updated {time.strftime('%I:%M:%S %p').lstrip('0')}")
        self.root.after(2000, self.refresh)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ProgressWindow().run()
