import torch
from .simple_mlp import SimpleMLP
from .cnn import TinyCNN
from .transformer import TinyTransformer

def make_demo(name):
    classes = {"mlp": (SimpleMLP, (2, 16)), "cnn": (TinyCNN, (2, 3, 16, 16)),
               "transformer": (TinyTransformer, (2, 16, 64))}
    cls, shape = classes[name]
    return cls().eval(), (torch.randn(*shape),)
