from ultralytics import YOLO
m = YOLO(r"C:\Users\Client\Downloads\FN AL tracker\models\al\best.pt")
print("AL names", m.names)
try:
    fn = YOLO(r"C:\Users\Client\Downloads\FN AL tracker\models\fn\best.pt")
    print("FN names", fn.names)
except Exception as e:
    print("FN", e)

import numpy as np, time, onnxruntime as ort
sess = ort.InferenceSession(
    r"C:\Users\Client\Downloads\FN AL tracker\models\al\best.fp16.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("providers", sess.get_providers())
inp = sess.get_inputs()[0]
print("input", inp.name, inp.shape, inp.type)
out = sess.get_outputs()[0]
print("output", out.name, out.shape, out.type)
x = np.random.rand(1, 3, 320, 320).astype(np.float32)
for _ in range(5):
    y = sess.run(None, {inp.name: x})[0]
t0 = time.perf_counter()
n = 50
for _ in range(n):
    y = sess.run(None, {inp.name: x})[0]
ms = (time.perf_counter() - t0) / n * 1000
print("shape", y.shape, "maxconf", float(y[0,:,4].max()) if y.size else None, f"{ms:.2f} ms", f"{1000/ms:.0f} Hz")
print(y[0, :3])
