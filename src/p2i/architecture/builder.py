from copy import deepcopy
import hashlib
import torch
from torch import nn
from .schema import ArchitectureIR,WeightTransferReport
from .registry import registry as default_registry

class BuildError(ValueError):
    def __init__(self,message,node=None):super().__init__(message);self.node=node

def build_architecture(architecture,*,originals=None,registry=None):
    arch=ArchitectureIR.model_validate(architecture.model_dump() if isinstance(architecture,ArchitectureIR) else architecture)
    originals=originals or {};registry=registry or default_registry
    nodes={n.id:n for n in arch.nodes};memo={};report=WeightTransferReport();paths={}
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    def construct(nid,path):
        paths[path]=nid
        if nid in memo:return memo[nid]
        n=nodes[nid];old=originals.get(nid);adapter=registry.adapters.get(n.module_type)
        try:
            if adapter:
                adapter.validate(n.parameters)
                seed=int.from_bytes(hashlib.sha256(nid.encode()).digest()[:4],'big')+arch.seed
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    module=adapter.build(deepcopy(n.parameters))
                reference=next(iter(old.parameters()),None) if old is not None else None
                if reference is None and old is not None:reference=next(iter(old.buffers()),None)
                if reference is not None:
                    module.to(device=reference.device)
                    if reference.is_floating_point() or reference.is_complex():module.to(dtype=reference.dtype)
            elif old is not None and n.module_type==type(old).__module__+'.'+type(old).__qualname__:
                module=deepcopy(old)
            else:raise ValueError('Custom module requires its live Harness session or a registered reconstruction adapter')
            memo[nid]=module
            if not adapter or not adapter.owns_children:
                if adapter and not adapter.container and n.children:raise ValueError('Leaf adapter cannot accept registered children')
                null_children={k:v for k,v in module._modules.items() if v is None} if not adapter else {}
                module._modules.clear()
                module._modules.update(null_children)
                for child in n.children:module.add_module(child.name,construct(child.node_id,f'{path}.{child.name}' if path else child.name))
            else:
                def check_owned(node_id,built):
                    recorded=nodes[node_id]
                    if set(r.name for r in recorded.children)!=set(k for k,v in built._modules.items() if v is not None):
                        raise ValueError('Adapter-owned registrations must match its constructed internals')
                    for registration in recorded.children:
                        child=nodes[registration.node_id]
                        if child.managed_by!=nid or child.editable:
                            raise ValueError('Adapter-owned child must be marked read-only and managed by its parent')
                        check_owned(child.id,built._modules[registration.name])
                # New adapter replacements may omit internal nodes until observed.
                if n.children:check_owned(nid,module)
                for subpath,submodule in module.named_modules():
                    if not subpath:continue
                    childpath=f'{path}.{subpath}' if path else subpath
                    match=next((x for x in arch.nodes if x.qualified_name==childpath and x.managed_by==nid),None)
                    if match:
                        paths[childpath]=match.id;memo[match.id]=submodule;submodule.training=match.training
            module.training=n.training
            # Module-owned state; child state transfers independently by stable node ID.
            recurse=bool(adapter and adapter.owns_children)
            newstate=module.state_dict()
            oldstate=old.state_dict() if old is not None else {}
            keys=[k for k in newstate if recurse or '.' not in k]
            with torch.no_grad():
                for k in keys:
                    label=f'{nid}:{k}'
                    if k in oldstate and oldstate[k].shape==newstate[k].shape:
                        newstate[k].copy_(oldstate[k]);report.preserved.append(label)
                    else:report.reinitialized.append(label)
                for k in oldstate:
                    if (recurse or '.' not in k) and (k not in newstate or oldstate[k].shape!=newstate[k].shape):report.dropped.append(f'{nid}:{k}')
            if old is not None:
                oldparams=dict(old.named_parameters(recurse=recurse))
                for key,param in module.named_parameters(recurse=recurse):
                    if key in oldparams:param.requires_grad_(oldparams[key].requires_grad)
            return module
        except Exception as exc:
            if isinstance(exc,BuildError):raise
            raise BuildError(str(exc),nid) from exc
    model=construct(arch.root_id,'')
    # Preserve explicitly tied parameter objects across distinct module instances.
    ties={}
    for nid,old in originals.items():
        if nid not in memo:continue
        for key,value in old.named_parameters(recurse=False):ties.setdefault(id(value),[]).append((nid,key))
    for entries in ties.values():
        if len(entries)<2:continue
        available=[(memo[nid],key) for nid,key in entries if key in memo[nid]._parameters and memo[nid]._parameters[key] is not None]
        if len(available)<2:continue
        first=available[0][0]._parameters[available[0][1]]
        for module,key in available[1:]:
            if module._parameters[key].shape!=first.shape:raise BuildError('Edit would break a tied parameter; coordinate all tied dimensions')
            module._parameters[key]=first
    for nid,old in originals.items():
        if nid not in memo:report.dropped.extend(f'{nid}:{key}' for key in old.state_dict() if '.' not in key)
    return model,report,paths

def build(architecture,*,registry=None):
    """Standalone constructor build, with deterministic new weights. Custom nodes need adapters."""
    return build_architecture(architecture,registry=registry)[0]
