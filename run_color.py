"""Quick color-based tracker — no model, <1ms detection."""
import ctypes
import time
import threading

# 1ms timer resolution
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

from exoai_tracker.color_pipeline import ColorPipeline

def main():
    print("=" * 50)
    print("COLOR TRACKER - No neural network, <1ms detection")
    print("=" * 50)
    print()
    print("Settings:")
    print("  Bind: Right Mouse Button (RMB)")
    print("  Color: RED (Apex enemy outlines)")
    print("  FOV: 150px radius")
    print()
    
    pipe = ColorPipeline(
        bind="right",
        speed=2.5,  # Smoother tracking
        fov=150,
        capture_size=320,
        capture_method="dxcam",
        color_preset="red",  # Red for Apex
        offset_y=0.45,
    )
    
    print("Initializing Razer mouse...")
    try:
        pipe.start()
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure Razer Synapse is running and you're admin!")
        input("Press Enter to exit...")
        return
    
    print(f"Mouse: {pipe.mouse.label}")
    print()
    print("READY! Hold RMB to track red targets.")
    print("Press Ctrl+C to stop.")
    print()
    
    try:
        while True:
            s = pipe.stats
            if s.state != "IDLE":
                print(f"\r{s.detail}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        pipe.stop()
        print("Done.")

if __name__ == "__main__":
    main()
