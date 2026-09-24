import inspect
import torch
from torch.fx.passes.shape_prop import ShapeProp

def capture(model, args, kwargs):
    gm = torch.fx.symbolic_trace(model)
    try:
        bound = inspect.signature(gm.forward).bind(*args, **kwargs)
        bound.apply_defaults()
        with torch.no_grad():
            ShapeProp(gm).propagate(*bound.arguments.values())
    except Exception as exc:
        # Keep the graph and any metadata that propagation completed.
        gm.meta['p2i_shape_error'] = f'{type(exc).__name__}: {exc}'[:1000]
    return gm
