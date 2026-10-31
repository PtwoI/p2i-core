# One complete P2I workflow

P2I 0.4.1 supports inspecting a model, editing registered modules, validating
and retracing the candidate, saving its current weights, and returning to an
ordinary trainable PyTorch module. It is a local workflow; no AI provider is
required.

## Install the split repositories

Use Python 3.11 or 3.12. For a CPU-only environment, install a supported CPU
PyTorch build first. From the directory that will contain the four checkouts:

```bash
git clone https://github.com/PtwoI/p2i-core.git
git clone https://github.com/PtwoI/p2i-gui.git
git clone https://github.com/PtwoI/p2i-sdk.git
git clone https://github.com/PtwoI/p2i-cli.git
python -m pip install -e ./p2i-core -e ./p2i-gui -e ./p2i-sdk -e ./p2i-cli
python -m p2i_cli demo transformer --edit
```

Open `http://127.0.0.1:8000`. Built frontend assets ship in `p2i-gui`, so
Node.js is only needed when developing the UI.

For a coordinated feature branch, check out that branch in all four
repositories before installing them. The CLI and GUI need core 0.4.1 or later.

## Change a model in the browser

1. Explore `blocks → blocks.0 → mlp` in the Transformer demo.
2. Open Edit, expand **Structured action JSON**, and paste this batch:

```json
{
  "type": "batch",
  "actions": [
    {"type": "set_parameter", "target": "blocks.0.mlp.0", "parameter": "out_features", "value": 96},
    {"type": "set_parameter", "target": "blocks.0.mlp.2", "parameter": "in_features", "value": 96}
  ]
}
```

3. Click **Preview & validate**, then **Commit edit**.
4. Click **Build & Re-trace** to refresh the observation.
5. Use **Download checkpoint** to save the edited model. Include examples if
   you plan to reopen it in the browser.
6. Make another edit or undo, then use **Open checkpoint** to restore the file.
   A new history starts at the saved revision; the previous unsaved session is
   replaced only after the file successfully restores and retraces.

Both explicitly connected dimensions change together. Compatible state is
copied exactly; changed tensor shapes are deterministically reinitialized.
The weight transfer report lists affected state. P2I does not promise that a
structurally valid change preserves task accuracy.

## Resume from a terminal or SDK

```bash
python -m p2i_cli demo transformer --no-serve --checkpoint original.p2i
python -m p2i_cli checkpoint inspect original.p2i
python -m p2i_cli checkpoint apply original.p2i action.json --output edited.p2i
python -m p2i_cli checkpoint validate edited.p2i --backward
python -m p2i_cli open edited.p2i
```

`action.json` can contain the batch above. Existing `p2i harness` commands
continue to operate on structure-only JSON files. Use `checkpoint` commands
when weights and examples must survive between processes.

```python
from p2i_sdk import Client

client = Client(timeout=60)
client.save_checkpoint("edited.p2i", expected_revision=1)
client.open_checkpoint("edited.p2i")
```

## Use your own Python model

```python
import torch
from my_project.models import MyModel
import p2i

model = MyModel().eval()
inputs = (torch.randn(2, 16),)
harness = p2i.Harness(model, example_inputs=inputs)
p2i.serve(harness)
```

The original forward implementation is kept in the live session. Standard
children can be edited; unknown constructors remain read-only. When restoring
in a new process, explicitly provide that implementation:

```python
harness = p2i.Harness.load("edited.p2i", model=MyModel())
model = harness.build()
```

Use the same source model and non-tensor configuration used to create the
session. A compatible state dictionary alone cannot reconstruct an arbitrary
Python forward. Checkpoints retain a mapping to the original template's
module paths, so supported child replacements and container edits can be
reapplied. Registered reconstruction adapters are the alternative for models
that must load independently. Files never supply their own import statements.

## Verification

The core tests cover exact weights and outputs after a round trip, an atomic
dimension edit, ordinary optimizer steps, templates with keyword arguments,
double precision, frozen parameters, tied parameters and shared modules,
per-module modes, and omitted inputs. GUI tests cover failed-import rollback
and a real browser download/upload flow for all three demos. CLI tests use
fresh subprocesses to save, edit, validate and reopen a checkpoint.
