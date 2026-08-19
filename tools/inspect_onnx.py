import onnx, os
from pathlib import Path
root = Path(r"C:\Users\Client\Downloads\FN AL tracker\models\al")
for p in sorted(root.glob("*.onnx")):
    m = onnx.load(str(p))
    print(p.name, p.stat().st_size)
    for i in m.graph.input:
        print("  in ", i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
    for o in m.graph.output:
        print("  out", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
