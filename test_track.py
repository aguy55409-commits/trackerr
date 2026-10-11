"""Test detection + mouse movement together."""
import ctypes
import time
from exoai_tracker import driver_mouse
from exoai_tracker.detector import YoloDetector
from exoai_tracker.models import resolve_preset_weights

user32 = ctypes.windll.user32
VK_RBUTTON = 0x02
VK_XBUTTON1 = 0x05  # Mouse4
VK_XBUTTON2 = 0x06  # Mouse5
VK_SHIFT = 0x10

def get_held_buttons():
    held = []
    if user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000:
        held.append("RMB")
    if user32.GetAsyncKeyState(0x01) & 0x8000:
        held.append("LMB")
    if user32.GetAsyncKeyState(VK_XBUTTON1) & 0x8000:
        held.append("Mouse4")
    if user32.GetAsyncKeyState(VK_XBUTTON2) & 0x8000:
        held.append("Mouse5")
    if user32.GetAsyncKeyState(VK_SHIFT) & 0x8000:
        held.append("Shift")
    return held

print("=== DETECTION + TRACKING TEST v2 ===")
print()

# Init mouse
print("Loading mouse driver...")
driver_mouse.init()
print(f"Mouse ready: {driver_mouse.ready()}")
print()

# Load model
print("Loading model...")
model_path = resolve_preset_weights("AL", "")
print(f"Model: {model_path}")

detector = YoloDetector(
    model_path=str(model_path),
    confidence=0.35,
    capture_size=320,
)
print(f"Screen: {detector.screen_w}x{detector.screen_h}")
print(f"Center: {detector.center_x}, {detector.center_y}")
print()

print("READY!")
print("Hold RIGHT MOUSE (or Mouse4/Mouse5) when looking at enemy.")
print("Will print status every frame.")
print()

cx = detector.center_x
cy = detector.center_y
speed = 0.3
frame_count = 0

try:
    while True:
        frame_count += 1
        held = get_held_buttons()
        
        # Always detect
        raw, _, _ = detector.detect(300, wait_for_new=False)
        
        # Print status every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: held={held}, targets={len(raw) if raw else 0}")
        
        # Track if any button held and targets exist
        if held and raw:
            # Pick closest
            best = min(raw, key=lambda d: (d.x - cx)**2 + (d.y - cy)**2)
            
            # Calculate movement  
            err_x = best.x - cx
            err_y = (best.y - best.height * 0.35) - cy
            
            dx = int(err_x * speed)
            dy = int(err_y * speed)
            
            # Clamp
            dx = max(-100, min(100, dx))
            dy = max(-100, min(100, dy))
            
            print(f"TARGET: err=({err_x:.0f},{err_y:.0f}) -> move=({dx},{dy}) conf={best.confidence:.2f}")
            
            if dx != 0 or dy != 0:
                # Use mouse_event directly
                ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)
                print(f"  >>> SENT MOVE: {dx}, {dy}")
        
        time.sleep(0.016)  # ~60hz to reduce FPS impact

except KeyboardInterrupt:
    print("\nStopped")
finally:
    detector.close()
