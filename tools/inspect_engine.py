import tensorrt as trt
from pathlib import Path

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
root = Path(r"C:\Users\Client\Downloads\FN AL tracker\models\al")
for p in [root / "fast.engine", root / "best.engine"]:
    print("\n==", p.name, p.stat().st_size)
    engine = runtime.deserialize_cuda_engine(p.read_bytes())
    n = engine.num_io_tensors
    for i in range(n):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(i, name, shape, dtype, mode)
