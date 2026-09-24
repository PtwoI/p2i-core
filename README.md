# p2i-core

Local-first PyTorch tracing, observed `ModelIR`, editable `ArchitectureIR`,
`Harness`, deterministic reconstruction, and the skill registry.

Install from this checkout with `python -m pip install -e '.[test]'`. This
distribution owns the `p2i` import. Basic usage:

```python
import p2i
ir = p2i.trace(model, example_inputs=(x,))
ir.save("model.json")
```

`p2i.serve(ir)` remains available when the separate `p2i-gui` package is
installed. Runtime tracing, actions, adapters, skill schemas, the in-process
Harness, and serialized IR remain in core. Frontend assets, FastAPI, the
terminal entry point, and any external AI integration are separate optional
repositories. Core never imports those packages during normal tracing.

Run `python -m pytest -q` after installation. See
[PtwoI/p2i](https://github.com/PtwoI/p2i) for the original project history.
