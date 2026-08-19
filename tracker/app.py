from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import webview

from exoai_tracker.binds import format_bind, is_bind_held
from tracker import resolve_engine
from tracker.aim import Aimer
from tracker.capture import ScreenCapture
from tracker.engine import TensorRTDetector

CONFIG_PATH = Path(__file__).resolve().parent.parent / "tracker_config.json"


@dataclass
class Config:
    preset: str = "AL"
    bind: str = "right"
    speed: float = 3.2
    fov: int = 120
    conf: float = 0.45
    capture: int = 320
    offset_y: float = 0.55

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})
        return cls()


HTML = """<!doctype html>
<html><head><meta charset="utf-8"/>
<style>
  :root { color-scheme: dark; font-family: Segoe UI, system-ui, sans-serif; }
  body { margin: 0; padding: 16px; background: #12141a; color: #e8eaef; }
  h1 { font-size: 15px; margin: 0 0 12px; font-weight: 600; letter-spacing: .02em; }
  label { display:block; font-size: 11px; opacity:.7; margin: 10px 0 4px; }
  select, input { width: 100%; box-sizing: border-box; background:#1c2030; border:1px solid #2a3144;
    color:#e8eaef; border-radius:6px; padding:8px 10px; }
  .row { display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
  button { margin-top: 14px; width:100%; padding:10px; border:0; border-radius:8px; font-weight:600; cursor:pointer; }
  #go { background:#3d7eff; color:#fff; }
  #go.running { background:#c44; }
  #stats { margin-top:14px; font-size:12px; line-height:1.5; opacity:.9; white-space:pre-wrap;
    background:#0c0e14; border-radius:8px; padding:10px; min-height:72px; }
</style></head><body>
<h1>TensorRT Tracker</h1>
<div class="row">
  <div><label>Model</label>
    <select id="preset"><option>AL</option><option>FN</option><option>VAL</option><option>R6</option></select>
  </div>
  <div><label>Bind</label>
    <select id="bind">
      <option value="right">RMB</option><option value="left">LMB</option>
      <option value="x1">Mouse4</option><option value="x2">Mouse5</option>
      <option value="shift">Shift</option>
    </select>
  </div>
</div>
<div class="row">
  <div><label>Speed</label><input id="speed" type="number" min="0.1" max="5" step="0.1" value="3.2"/></div>
  <div><label>FOV</label><input id="fov" type="number" min="40" max="400" step="1" value="120"/></div>
</div>
<div class="row">
  <div><label>Confidence</label><input id="conf" type="number" min="0.1" max="0.9" step="0.05" value="0.45"/></div>
  <div><label>Capture</label><input id="capture" type="number" min="160" max="640" step="32" value="320"/></div>
</div>
<button id="go">Start</button>
<div id="stats">Ready</div>
<script>
async function refresh(){
  const s = await pywebview.api.stats();
  document.getElementById('stats').textContent = s.text;
  const b = document.getElementById('go');
  b.textContent = s.running ? 'Stop' : 'Start';
  b.className = s.running ? 'running' : '';
}
async function boot(){
  const c = await pywebview.api.get_config();
  for (const k of ['preset','bind','speed','fov','conf','capture']) {
    const el = document.getElementById(k); if (el && c[k] != null) el.value = c[k];
  }
  setInterval(refresh, 200);
}
document.getElementById('go').onclick = async () => {
  const cfg = {
    preset: document.getElementById('preset').value,
    bind: document.getElementById('bind').value,
    speed: parseFloat(document.getElementById('speed').value),
    fov: parseInt(document.getElementById('fov').value),
    conf: parseFloat(document.getElementById('conf').value),
    capture: parseInt(document.getElementById('capture').value),
  };
  await pywebview.api.toggle(cfg);
  refresh();
};
window.addEventListener('pywebviewready', boot);
</script>
</body></html>
"""


class Api:
    def __init__(self, app: "App") -> None:
        self.app = app

    def get_config(self) -> dict:
        return asdict(self.app.cfg)

    def stats(self) -> dict:
        return {"text": self.app.status_text(), "running": self.app.running}

    def toggle(self, cfg: dict) -> None:
        self.app.apply(cfg)
        if self.app.running:
            self.app.stop()
        else:
            self.app.start()


class App:
    def __init__(self) -> None:
        self.cfg = Config.load()
        self.running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: ScreenCapture | None = None
        self._det: TensorRTDetector | None = None
        self.aimer = Aimer()
        self._hz = 0.0
        self._ms = 0.0
        self._state = "IDLE"
        self._detail = "Ready"
        self._api = Api(self)

    def apply(self, data: dict) -> None:
        self.cfg.preset = str(data.get("preset", self.cfg.preset)).upper()
        self.cfg.bind = str(data.get("bind", self.cfg.bind))
        self.cfg.speed = float(data.get("speed", self.cfg.speed))
        self.cfg.fov = int(data.get("fov", self.cfg.fov))
        self.cfg.conf = float(data.get("conf", self.cfg.conf))
        self.cfg.capture = int(data.get("capture", self.cfg.capture))
        self.cfg.save()
        self.aimer.speed = self.cfg.speed
        self.aimer.fov = self.cfg.fov
        self.aimer.offset_y = self.cfg.offset_y

    def status_text(self) -> str:
        return (
            f"{self._state}  ·  {self._hz:.0f} Hz  ·  {self._ms:.1f} ms\n"
            f"mouse {self.aimer.mouse.label}  ·  move {self.aimer.last_dx},{self.aimer.last_dy}\n"
            f"{self._detail}"
        )

    def start(self) -> None:
        if self.running:
            return
        engine = resolve_engine(self.cfg.preset)
        self._detail = f"Loading {engine.name}…"
        self._cap = ScreenCapture(self.cfg.capture)
        self._cap.start()
        self._det = TensorRTDetector(engine, conf=self.cfg.conf, imgsz=self.cfg.capture)
        self.aimer = Aimer()
        self.aimer.speed = self.cfg.speed
        self.aimer.fov = self.cfg.fov
        self.aimer.offset_y = self.cfg.offset_y
        self._detail = f"{self.cfg.preset} · {engine.name} · {self.aimer.mouse.label} · hold [{format_bind(self.cfg.bind)}]"
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="track", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        if self._cap:
            self._cap.stop()
            self._cap = None
        self._det = None
        self._state = "IDLE"
        self._detail = "Stopped"
        self._hz = 0.0

    def _loop(self) -> None:
        assert self._cap and self._det
        last_seq = -1
        n = 0
        t0 = time.perf_counter()
        last = time.perf_counter()
        while not self._stop.is_set():
            frame = self._cap.latest()
            if frame is None or frame.seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = frame.seq
            t_infer = time.perf_counter()
            dets = self._det.predict(frame.rgb, frame.left, frame.top)
            self._ms = (time.perf_counter() - t_infer) * 1000.0
            now = time.perf_counter()
            dt = now - last
            last = now
            held = is_bind_held(self.cfg.bind)
            cx, cy = self._cap.center_x, self._cap.center_y
            target = self.aimer.pick(dets, cx, cy) if held else None
            if held and target is not None:
                self._state = "TRACK"
                self.aimer.step(target, cx, cy, dt)
            elif held and dets:
                self._state = "FOV"
            elif held:
                self._state = "SCAN"
            else:
                self._state = "IDLE"
            n += 1
            if now - t0 >= 0.5:
                self._hz = n / (now - t0)
                n = 0
                t0 = now

    def run(self) -> None:
        webview.create_window("TensorRT Tracker", html=HTML, js_api=self._api, width=380, height=460)
        webview.start()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
