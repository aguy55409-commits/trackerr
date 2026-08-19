# Models

Trained TensorRT engines and `.pt` weights are **not** in git (too large).

Local layout expected:

```
models/
  al/best.engine   # Apex
  fn/best.engine   # Fortnite
  val/best.engine  # Valorant
  r6/best.engine   # Rainbow Six
```

Keep your local `models/` folder backed up separately (copy/drive). Code in this repo loads those paths at runtime.
