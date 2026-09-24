import torch

def capture(model, args, kwargs, dynamic_shapes=None):
    return torch.export.export(model, args, kwargs, dynamic_shapes=dynamic_shapes, strict=False).graph_module
