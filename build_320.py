"""Build best_combined_320.engine from best_combined.onnx using ONLY the TensorRT
Python API. No modelopt, no pip, no torch needed (works even with CPU torch)."""
import os
import tensorrt as trt

ROOT = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(ROOT, "models", "al", "best_combined.onnx")
ENGINE = os.path.join(ROOT, "models", "al", "best_combined_320.engine")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def main() -> None:
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    # 4 GB workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    # Try FP32 first (no FP16 flag) to isolate the crash. TRT 11 has no FP16 flag attr.
    print("builder flags available:", [a for a in dir(trt.BuilderFlag) if not a.startswith('_')], flush=True)



    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(ONNX, "rb") as f:
        parser.parse(f.read())
    if parser.num_errors:
        for i in range(parser.num_errors):
            print("ONNX ERROR:", parser.get_error(i), flush=True)
        raise SystemExit(1)

    inp = network.get_input(0)
    # TRT shape tuple omits/keeps the batch dim depending on version.
    # Normalize to a 4-D (1,3,320,320) spec for the optimization profile.
    dims = list(inp.shape)  # may be (3,320,320) or (1,3,320,320)
    if len(dims) == 3:
        dims = [1] + dims
    spec = (1, 3, 320, 320)
    print("input", inp.name, "trt_shape", dims, "-> profile", spec, "dtype", inp.dtype, flush=True)
    try:
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, spec, spec, spec)
        config.add_optimization_profile(profile)
    except Exception as e:
        print("profile failed (fixed-shape ONNX; building without profile):", e, flush=True)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("build returned None")
    with open(ENGINE, "wb") as f:
        f.write(engine)
    print(f"WROTE {ENGINE} ({os.path.getsize(ENGINE)} bytes)", flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()

