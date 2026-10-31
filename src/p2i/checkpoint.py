"""Model round trips without pickling Python modules or importing artifact code."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from p2i.architecture.builder import build_architecture
from p2i.architecture.schema import ArchitectureIR


def _portable(value):
    """Only tensors and ordinary containers may enter the checkpoint."""
    if isinstance(value, torch.Tensor):
        if type(value) not in (torch.Tensor, nn.Parameter):
            raise ValueError('Checkpoint inputs must use ordinary tensors')
        return value.detach().cpu().clone()
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) in (list, tuple):
        return type(value)(_portable(v) for v in value)
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _portable(v) for k, v in value.items()}
    raise ValueError(f'Unsupported checkpoint value: {type(value).__name__}')


def _builtin_spec(module):
    # Explicit trusted demo implementations, never an import path from a file.
    from p2i.examples.simple_mlp import SimpleMLP
    from p2i.examples.cnn import TinyCNN
    from p2i.examples.transformer import TinyTransformer, TransformerBlock, SelfAttention
    if type(module) is SimpleMLP:
        return {'name': 'mlp', 'parameters': {}}
    if type(module) is TinyCNN:
        return {'name': 'cnn', 'parameters': {}}
    if type(module) is TinyTransformer:
        return {'name': 'transformer', 'parameters': {'width': module.norm.normalized_shape[0], 'depth': len(module.blocks)}}
    if type(module) is TransformerBlock:
        return {'name': 'block', 'parameters': {'width': module.norm1.normalized_shape[0]}}
    if type(module) is SelfAttention:
        return {'name': 'attention', 'parameters': {'width': module.qkv.in_features, 'heads': module.heads}}
    return None


def _builtin_model(spec):
    from p2i.examples.simple_mlp import SimpleMLP
    from p2i.examples.cnn import TinyCNN
    from p2i.examples.transformer import TinyTransformer, TransformerBlock, SelfAttention
    factories = {'mlp': SimpleMLP, 'cnn': TinyCNN, 'transformer': TinyTransformer,
                 'block': TransformerBlock, 'attention': SelfAttention}
    if spec['name'] not in factories:
        raise ValueError('Unknown built-in checkpoint template')
    return factories[spec['name']](**spec['parameters'])


def _aliases(entries):
    groups = {}
    for name, value in entries:
        groups.setdefault(id(value), []).append(name)
    return [names for names in groups.values() if len(names) > 1]


def checkpoint_bytes(harness, *, include_inputs=True):
    """Serialize the current architecture, weights, modes, and optional examples.

    A checkpoint is a new editing baseline, not an optimizer or undo-history save.
    Unknown custom forward implementations require a caller-supplied template on
    load; built-in demos and registered adapters are independently reconstructible.
    """
    with harness._lock:
        model = harness.build()
        source_paths = {id(m): p for p, m in harness._model.named_modules()} if harness._model is not None else {}
        templates = {}
        for node in harness._arch.nodes:
            if node.module_type in harness.registry.adapters:
                continue
            original = harness._originals.get(node.id)
            if original is None:
                raise ValueError(f'No reconstruction template for {node.qualified_name or "(root)"}')
            templates[node.id] = {'source_path': source_paths.get(id(original)),
                                  'builtin': _builtin_spec(original)}
        state = model.state_dict()
        if not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError('Custom get_extra_state requires a tensor-only reconstruction adapter')
        payload = {
            'format': 'p2i.checkpoint', 'version': 1,
            'architecture': harness.architecture().model_dump(mode='json'),
            'state_dict': {k: _portable(v) for k, v in state.items()},
            'templates': templates,
            'requires_grad': {k: v.requires_grad for k, v in model.named_parameters(remove_duplicate=False)},
            'parameter_aliases': _aliases(model.named_parameters(remove_duplicate=False)),
            'buffers': {k: _portable(v) for k, v in model.named_buffers(remove_duplicate=False)},
            'buffer_aliases': _aliases(model.named_buffers(remove_duplicate=False)),
            'example_inputs': _portable(harness._args) if include_inputs else None,
            'example_kwargs': _portable(harness._kwargs) if include_inputs else {},
        }
        stream = BytesIO()
        torch.save(payload, stream)
        return stream.getvalue()


def save_checkpoint(harness, path, *, include_inputs=True):
    """Atomically replace a checkpoint only after serialization succeeds."""
    data = checkpoint_bytes(harness, include_inputs=include_inputs)
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if type(value) in (list, tuple):
        return type(value)(_move(v, device) for v in value)
    if type(value) is dict:
        return {k: _move(v, device) for k, v in value.items()}
    return value


def _retie(model, groups, attribute):
    for names in groups:
        first = None
        for name in names:
            parent_path, _, key = name.rpartition('.')
            parent = model.get_submodule(parent_path)
            values = getattr(parent, attribute)
            if key not in values or values[key] is None:
                raise ValueError(f'Invalid checkpoint alias: {name}')
            if first is None:
                first = values[key]
            else:
                if not torch.equal(first, values[key]) or first.dtype != values[key].dtype:
                    raise ValueError(f'Conflicting checkpoint alias: {name}')
                values[key] = first


def load_checkpoint(source, *, model=None, registry=None, skill_registry=None, device='cpu'):
    """Restore a Harness on CPU by default using PyTorch's weights-only loader.

    Supply ``model=YourModel()`` for unregistered custom Python forwards. This
    template supplies code and non-tensor attributes, while checkpoint structure
    and weights replace its registered children and state. It must be the same
    source architecture used to create the checkpoint, with matching attributes.
    """
    from p2i.harness import Harness
    from p2i.architecture.registry import registry as default_registry
    try:
        stream = BytesIO(source) if isinstance(source, bytes) else source
        payload = torch.load(stream, map_location='cpu', weights_only=True)
    except (OSError, FileNotFoundError):
        raise
    except Exception as exc:
        raise ValueError('Cannot read a weights-only P2I checkpoint') from exc
    if not isinstance(payload, dict) or payload.get('format') != 'p2i.checkpoint' or payload.get('version') != 1:
        raise ValueError('Not a supported P2I checkpoint (expected version 1)')
    arch = ArchitectureIR.model_validate(payload['architecture'])
    active_registry = (registry or default_registry).copy()
    # Resolve explicitly pinned skills through the caller's registry, if any.
    obj = Harness.from_architecture(arch, registry=active_registry, skill_registry=skill_registry)
    originals = {}
    template = deepcopy(model) if model is not None else None
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        for node in arch.nodes:
            if node.module_type in obj.registry.adapters:
                continue
            spec = payload['templates'].get(node.id)
            if spec is None:
                raise ValueError(f'Missing reconstruction template for {node.qualified_name or "(root)"}')
            if spec['builtin'] is not None:
                original = _builtin_model(spec['builtin'])
            elif template is not None and spec['source_path'] is not None:
                try:
                    original = template.get_submodule(spec['source_path'])
                except AttributeError as exc:
                    raise ValueError(f'Template is missing module {spec["source_path"]!r}') from exc
            else:
                raise ValueError(f'Custom module {node.module_type} requires model=YourModel() or a registered adapter')
            if f'{type(original).__module__}.{type(original).__qualname__}' != node.module_type:
                raise ValueError(f'Template type mismatch at {node.qualified_name or "(root)"}')
            originals[node.id] = original
        rebuilt, _, paths = build_architecture(arch, originals=originals, registry=obj.registry)
    # assign retains tensor dtypes; ties are explicitly restored after assignment.
    rebuilt.load_state_dict(payload['state_dict'], strict=True, assign=True)
    for name, value in payload['buffers'].items():
        parent_path, _, key = name.rpartition('.')
        parent = rebuilt.get_submodule(parent_path)
        if key not in parent._buffers:
            raise ValueError(f'Template is missing buffer {name}')
        parent._buffers[key] = value
    for name, parameter in rebuilt.named_parameters(remove_duplicate=False):
        parameter.requires_grad_(payload['requires_grad'][name])
    rebuilt.to(device)
    _retie(rebuilt, payload['parameter_aliases'], '_parameters')
    _retie(rebuilt, payload['buffer_aliases'], '_buffers')
    obj._model = rebuilt
    obj._originals = {nid: rebuilt.get_submodule(path) for path, nid in paths.items()}
    obj._args = _move(payload['example_inputs'], device)
    obj._kwargs = _move(payload['example_kwargs'], device)
    # No stale runtime evidence or tensor contracts survive the new baseline.
    for node in obj._arch.nodes:
        node.latest_observed_module_id = None
        node.input_contracts = []
        node.output_contracts = []
        node.contracts_revision = None
    obj._init_session(None)
    obj._revisions = {arch.revision: obj._arch.model_copy(deep=True)}
    return obj


def load_model(source, **options):
    """Return an ordinary trainable nn.Module from a P2I checkpoint."""
    return load_checkpoint(source, **options).build()
