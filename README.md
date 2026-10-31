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

## Edit, save, reopen, and train

```python
import torch
from torch import nn
import p2i

inputs = (torch.randn(2, 16),)
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
harness = p2i.Harness(model, example_inputs=inputs)
result = harness.apply({"type": "batch", "actions": [
    {"type": "set_parameter", "target": "0", "parameter": "out_features", "value": 24},
    {"type": "set_parameter", "target": "2", "parameter": "in_features", "value": 24},
]})
assert result.success, result.errors
assert harness.validate(backward=True).backward_valid
harness.retrace()
harness.save("model.p2i")

restored = p2i.Harness.load("model.p2i")
p2i.serve(restored)  # Optional GUI; omit to continue directly in Python.
```

The checkpoint contains architecture, weights, per-module train/eval modes,
parameter sharing, and example inputs. It uses `torch.load(weights_only=True)`;
it does not pickle a Python model or import source named by an artifact.
`ArchitectureIR.save()` remains a structure-only JSON export with newly
initialized weights when reconstructed independently.

```python
model = p2i.load_model("model.p2i")  # Ordinary nn.Module, CPU by default.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
optimizer.zero_grad()
loss = model(*inputs).square().mean()
loss.backward()
optimizer.step()

# Bring newly trained weights back into P2I.
p2i.Harness(model, example_inputs=inputs).save("trained.p2i")
```

`harness.build()` returns an independent model. Training that model does not
mutate the live session; start a new Harness with it to save the new weights.
See [the complete walkthrough](docs/model-roundtrip.md) and run
`python examples/model_roundtrip.py --output model.p2i` for an offline example.

## Supported first-release workflow

Standard registered modules and bundled MLP/CNN/Transformer demos round-trip
without a source model. Custom `nn.Module` forwards work in live sessions;
restore their checkpoints with `Harness.load(path, model=YourModel(...))` or
provide a reconstruction adapter through `registry=`. The template supplies
Python behavior and non-tensor attributes and must match the source model.
Custom skills require their pinned definitions in `skill_registry=`.

A checkpoint begins a new history at its saved revision. Optimizer state,
past undo/redo history, runtime captures, and arbitrary Python objects are not
included. Save with `include_inputs=False` when examples should be omitted;
the resulting model still loads for PyTorch use, but cannot be retraced until
you start a Harness with new example inputs. Use `device="cuda"` on load to
move a model and its examples together.

## Recurrent models and connected edits (0.4.2)

RNN, LSTM and GRU have constructor adapters, including stacked/bidirectional
variants, RNN nonlinearity and LSTM projection size. `make_demo("rnn")`,
`make_demo("lstm")` and `make_demo("gru")` provide offline classifiers with
independently restorable checkpoints. Recurrent state tuples are retained in
observations; editing a hidden width and its classifier input is an atomic batch.

```python
plan = harness.plan_edit("layers.0", "out_features", 24,
                         expected_revision=harness.architecture().revision)
print(plan["notes"], plan["unresolved"])
assert harness.preview(plan["action"]).success
harness.apply(plan["action"])
```

Suggestions follow explicit Sequential order through known dimension-preserving
modules, updating Linear inputs or Conv inputs and intervening BatchNorm sizes.
Shared consumers, flattening and arbitrary custom forwards require explicit
edits. The proposal does not mutate the session and still requires normal
preview/apply validation. Newly inserted modules inherit the source model's
reference device and dtype; heterogeneous device layouts require explicit adapters.

### External compatibility and GPU checks

```bash
python -m pip install -e '.[test,compatibility]'
HF_HUB_OFFLINE=1 python -m pytest -q tests/test_compatibility.py
P2I_TEST_DEVICE=cuda HF_HUB_OFFLINE=1 python -m pytest -q tests/test_compatibility.py
```

The suite uses TorchVision ResNet18 and MobileNetV3-Small, and a small configured
Hugging Face BERT, all with random weights and no model downloads. It verifies
runtime capture, a classifier edit, exact checkpoint restoration with the source
template, backward validation and a real optimizer step. This checks library
compatibility, not pretrained accuracy or large-model memory scalability.

CPU compatibility runs automatically in CI. CUDA checks additionally exercise
recurrent round trips and device inheritance for inserted modules. Run them on a
CUDA-capable machine, or select `cuda` when manually dispatching the external
compatibility workflow on a self-hosted runner labeled `linux`, `x64`, `gpu`.
An explicit CUDA request fails if no GPU is available. CPU CI success does not
indicate a GPU run. The development environment used for this change has no CUDA
GPU, so its GPU verification remains to be executed on such a runner.
