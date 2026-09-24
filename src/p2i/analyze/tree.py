"""Flatten nested tensor containers without evaluating arbitrary properties."""
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
import torch

def tensors(tree):
    seen = set()
    def walk(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, Mapping):
            values = obj.values()
        elif isinstance(obj, (tuple, list)):
            values = obj
        elif is_dataclass(obj) and not isinstance(obj, type):
            values = (getattr(obj, f.name) for f in fields(obj))
        else:
            return
        for value in values:
            yield from walk(value)
    return list(walk(tree))
