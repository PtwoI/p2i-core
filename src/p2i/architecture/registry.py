"""Explicit module-level adapter allowlist. JSON never imports arbitrary classes."""
from copy import deepcopy
import math
from typing import Any
import torch
from torch import nn

class ModuleAdapter:
    """User extension: implement extract and build; validation must raise ValueError."""
    module_type: str
    module_class: type[nn.Module]
    owns_children = False
    container = False
    def matches(self,module):return type(module) is self.module_class
    def extract(self,module):raise NotImplementedError
    def validate(self,parameters):pass
    def build(self,parameters):raise NotImplementedError

class ModuleAdapterRegistry:
    def __init__(self):self.adapters={}
    def register(self,module_type,adapter):
        if isinstance(adapter,type):adapter=adapter()
        adapter.module_type=module_type
        self.adapters[module_type]=adapter
        return adapter
    def find(self,module):
        return next((a for a in self.adapters.values() if a.matches(module)),None)
    def get(self,module_type):
        if module_type not in self.adapters:raise ValueError(f'No registered adapter for {module_type}')
        return self.adapters[module_type]
    def copy(self):
        registry=ModuleAdapterRegistry();registry.adapters=self.adapters.copy();return registry

class StandardAdapter(ModuleAdapter):
    def __init__(self,cls,fields,*,container=False,owns_children=False):
        self.module_class=cls;self.fields=fields;self.container=container;self.owns_children=owns_children
    def extract(self,module):
        values={}
        for key in self.fields:
            if key=='bias' and isinstance(module,nn.MultiheadAttention):value=module.in_proj_bias is not None
            elif key=='add_bias_kv':value=module.bias_k is not None
            elif key=='bias' and isinstance(module,(nn.Linear,nn.Conv1d,nn.Conv2d,nn.LayerNorm)):value=module.bias is not None
            else:value=getattr(module,key)
            if isinstance(value,tuple):value=list(value)
            values[key]=value
        return values
    def validate(self,p):
        extra=set(p)-set(self.fields)
        if extra:raise ValueError(f'Unsupported constructor parameters: {sorted(extra)}')
        positive={'in_features','out_features','in_channels','out_channels','groups','num_embeddings','embedding_dim','embed_dim','num_heads','num_features','kdim','vdim'}
        boolean={'bias','inplace','elementwise_affine','affine','track_running_stats','batch_first','add_bias_kv','add_zero_attn','scale_grad_by_freq','sparse','ceil_mode','return_indices','count_include_pad'}
        for k,v in p.items():
            if k in positive and v is not None and (type(v) is not int or v<=0):raise ValueError(f'{k} must be a positive integer')
            if k in boolean and type(v) is not bool:raise ValueError(f'{k} must be boolean')
            if k in {'p','dropout','eps','momentum','max_norm','norm_type'} and v is not None:
                if type(v) not in (int,float) or not math.isfinite(v):raise ValueError(f'{k} must be finite numeric')
                if k in {'p','dropout','momentum'} and not 0<=v<=1:raise ValueError(f'{k} must lie in [0, 1]')
                if k in {'eps','max_norm','norm_type'} and v<=0:raise ValueError(f'{k} must be positive')
            if k in {'kernel_size','stride','dilation','normalized_shape','output_size'} and v is not None:
                values=v if isinstance(v,(tuple,list)) else [v]
                if not values or any(x is not None and (type(x) is not int or x<=0) for x in values):raise ValueError(f'{k} must contain positive integers')
            if k in {'start_dim','end_dim','padding_idx'} and v is not None and type(v) is not int:raise ValueError(f'{k} must be integer')
        if 'approximate' in p and p['approximate'] not in ('none','tanh'):raise ValueError('approximate must be none or tanh')
        if 'padding_mode' in p and p['padding_mode'] not in ('zeros','reflect','replicate','circular'):raise ValueError('Invalid padding_mode')
        if 'embed_dim' in p and 'num_heads' in p and p['embed_dim']%p['num_heads']:
            raise ValueError(f"Cannot set num_heads={p['num_heads']}: embed_dim={p['embed_dim']} is not divisible by it")
        if 'groups' in p and any(p.get(k,p['groups'])%p['groups'] for k in ('in_channels','out_channels')):raise ValueError('in_channels and out_channels must be divisible by groups')
    def build(self,p):
        self.validate(p)
        return self.module_class(**p)

registry=ModuleAdapterRegistry()
def _add(cls,fields='',**kw):registry.register('torch.nn.'+cls.__name__,StandardAdapter(cls,fields.split(),**kw))
_add(nn.Linear,'in_features out_features bias')
for cls in (nn.Conv1d,nn.Conv2d):_add(cls,'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode')
for cls in (nn.ReLU,nn.SiLU):_add(cls,'inplace')
_add(nn.GELU,'approximate');_add(nn.Dropout,'p inplace')
_add(nn.LayerNorm,'normalized_shape eps elementwise_affine bias')
for cls in (nn.BatchNorm1d,nn.BatchNorm2d):_add(cls,'num_features eps momentum affine track_running_stats')
_add(nn.Embedding,'num_embeddings embedding_dim padding_idx max_norm norm_type scale_grad_by_freq sparse')
_add(nn.MultiheadAttention,'embed_dim num_heads dropout bias add_bias_kv add_zero_attn kdim vdim batch_first',owns_children=True)
for cls in (nn.Sequential,nn.ModuleList):_add(cls,container=True)
_add(nn.Softmax,'dim')
_add(nn.Identity);_add(nn.Flatten,'start_dim end_dim')
_add(nn.MaxPool2d,'kernel_size stride padding dilation return_indices ceil_mode')
_add(nn.AvgPool2d,'kernel_size stride padding ceil_mode count_include_pad divisor_override')
_add(nn.AdaptiveAvgPool2d,'output_size')

def register_module_adapter(module_class,*,module_type=None):
    """Decorator registering a trusted, in-process adapter implementation."""
    def decorate(adapter_class):
        adapter=adapter_class();adapter.module_class=module_class
        registry.register(module_type or f'{module_class.__module__}.{module_class.__qualname__}',adapter)
        return adapter_class
    return decorate
