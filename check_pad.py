"""Run this, then press things on the controller. Results are written to
_pad_result.txt so it does not matter when it is run.

  python check_pad.py
"""
import time

from exoai_tracker.binds import is_bind_held
from exoai_tracker.config import load_config
from exoai_tracker.controller_output import get_real_controller_state

cfg = load_config()
bind = cfg.controller_bind

print("=" * 58)
print("  Press EVERY control: LT, RT, LB, RB, A/B/X/Y, both sticks.")
print("  You have 25 seconds. Go.")
print("=" * 58)

seen: set[str] = set()
bind_held = False
changes = 0
last = None
t_end = time.time() + 25

while time.time() < t_end:
    st = get_real_controller_state()
    if st is None:
        time.sleep(0.05)
        continue
    g = st.Gamepad
    snap = (g.wButtons, g.bLeftTrigger, g.bRightTrigger,
            g.sThumbLX // 3000, g.sThumbLY // 3000,
            g.sThumbRX // 3000, g.sThumbRY // 3000)
    if snap != last:
        changes += 1
        last = snap
        if g.bLeftTrigger > 30:
            seen.add("LT")
        if g.bRightTrigger > 30:
            seen.add("RT")
        if g.wButtons & 0x0100:
            seen.add("LB")
        if g.wButtons & 0x0200:
            seen.add("RB")
        if g.wButtons & 0xF000:
            seen.add("face buttons")
        if abs(g.sThumbLX) > 9000 or abs(g.sThumbLY) > 9000:
            seen.add("left stick")
        if abs(g.sThumbRX) > 9000 or abs(g.sThumbRY) > 9000:
            seen.add("right stick")
        remaining = int(t_end - time.time())
        print(f"  [{remaining:2d}s] LT={g.bLeftTrigger:<3} RT={g.bRightTrigger:<3} "
              f"btn={g.wButtons:#06x}  seen: {', '.join(sorted(seen)) or '-'}")
    if is_bind_held(bind):
        bind_held = True
    time.sleep(0.008)

lines = [
    f"state changes: {changes}",
    f"controls seen: {', '.join(sorted(seen)) if seen else 'NOTHING'}",
    f"configured bind {bind!r} registered as held: {bind_held}",
]
report = "\n".join(lines)
print("\n" + "=" * 58)
print(report)
print("=" * 58)

with open("_pad_result.txt", "w", encoding="utf-8") as fh:
    fh.write(report + "\n")
print("\nwritten to _pad_result.txt")
