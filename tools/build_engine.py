"""Build a fixed-shape TensorRT engine from an ONNX model."""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import tensorrt as trt
from onnxconverter_common import float16


def build(
    onnx_path: Path, output: Path, workspace_gib: float, fp16: bool = True
) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 10+ networks are explicit-batch by default.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    model = onnx.load(str(onnx_path))
    if fp16:
        model = float16.convert_float_to_float16(
            model, keep_io_types=False, disable_shape_infer=True
        )
    if not parser.parse(model.SerializeToString()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30))
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    output.write_bytes(bytes(serialized))
    print(f"Built {output} ({output.stat().st_size / (1 << 20):.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    build(args.onnx, args.output, args.workspace, fp16=not args.fp32)


if __name__ == "__main__":
    main()
