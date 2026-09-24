from copy import deepcopy
import torch
from p2i.ir import ModelIR, ModelMetadata, RuntimeMetadata, AnalysisStatus, DataEdge
from .modules import extract
from .runtime import Recorder
from .static import ingest
from . import export, fx

def trace(model, example_inputs=None, *, example_args=None, example_kwargs=None,
          static=True, dynamic_shapes=None, isolate=True, inspection=None):
    """Analyze one forward pass plus export (FX fallback).

    Training/eval mode is respected. Work on a deep copy by default; input
    tensors must be deepcopy-compatible (normally leaf tensors). Static and
    runtime graphs are separate observations. No tensor payloads are retained.
    ``isolate=False`` explicitly permits mutation of the supplied model/inputs.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("p2i.trace expects an instantiated torch.nn.Module")
    if example_inputs is not None and example_args is not None:
        raise ValueError("Use example_inputs or example_args, not both")
    args = example_args if example_args is not None else example_inputs
    args = () if args is None else args
    if not isinstance(args, tuple):
        raise TypeError("Example positional inputs must be a tuple, e.g. (x,)")
    kwargs = example_kwargs or {}
    if not isinstance(kwargs, dict):
        raise TypeError("example_kwargs must be a dict")
    try:
        working, runtime_args, runtime_kwargs = deepcopy((model, args, kwargs)) if isolate else (model, args, kwargs)
    except Exception as exc:
        raise ValueError("Model/inputs could not be copied. Use deepcopy-compatible inputs or explicitly set isolate=False (forward may mutate model and inputs).") from exc
    nodes, edges, identities, paths = extract(working)
    ir = ModelIR(metadata=ModelMetadata(name=type(model).__name__, torch_version=str(torch.__version__), training=model.training),
                 modules=nodes, module_edges=edges, runtime=RuntimeMetadata())
    def status(technique, result, message=None):
        ir.metadata.analysis.append(AnalysisStatus(technique=technique, status=result, message=message))
    status("hierarchy", "success")
    recorder = Recorder(ir, identities, inspection=inspection)
    # fork_rng restores torch random generators, including initialized CUDA devices.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:
            recorder.install(working)
            with torch.no_grad(), recorder:
                ir.runtime.input_tensor_ids = recorder.ids((runtime_args, runtime_kwargs))
                output = working(*runtime_args, **runtime_kwargs)
                ir.runtime.output_tensor_ids = recorder.ids(output)
            status("runtime_hooks", "success")
            status("runtime_dispatch", "success")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:2000]
            status("runtime_hooks", "partial" if ir.runtime.calls else "failed", message)
            status("runtime_dispatch", "partial" if ir.operations else "failed", message)
            ir.metadata.warnings.append("Forward failed; runtime contains only the observed execution prefix. Incomplete calls are marked.")
        finally:
            recorder.close()
        if static:
            succeeded = False
            for name, backend in (("export", export), ("fx", fx)):
                if succeeded:
                    status(name, "skipped", "Export succeeded; FX fallback was not needed.")
                    continue
                try:
                    smodel, sargs, skwargs = deepcopy((model, args, kwargs))
                    gm = backend.capture(smodel, sargs, skwargs, dynamic_shapes) if name == "export" else backend.capture(smodel, sargs, skwargs)
                    # Ingest transactionally; failure must not leave half a graph.
                    fragment = ModelIR(metadata=ir.metadata)
                    ingest(gm, fragment, paths, name)
                    ir.operations.extend(fragment.operations)
                    ir.tensors.extend(fragment.tensors)
                    shape_error = gm.meta.get("p2i_shape_error")
                    status(name, "partial" if shape_error else "success",
                           f"Graph captured; shape propagation failed: {shape_error}" if shape_error else
                           "Static graph; executed flags are not inferred." if name == "fx" else None)
                    succeeded = True
                except Exception as exc:
                    status(name, "failed", f"{type(exc).__name__}: {exc}"[:2000])
        else:
            status("export", "skipped", "static=False")
            status("fx", "skipped", "static=False")
    ir.data_edges = [DataEdge(tensor_id=t.id, source_id=t.producer_id, target_id=c)
                     for t in ir.tensors if t.producer_id for c in t.consumer_ids]
    return ModelIR.model_validate(ir.model_dump())
