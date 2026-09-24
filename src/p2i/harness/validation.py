from copy import deepcopy
import traceback
import torch
from p2i.analyze.tree import tensors
from p2i.architecture.schema import ArchitectureIR,ValidationReport,ValidationIssue
from p2i.architecture.builder import build_architecture,BuildError
from p2i.architecture.registry import registry as default_registry

def validate_architecture(arch,*,originals=None,registry=None,args=None,kwargs=None,backward=False):
    report=ValidationReport(revision=getattr(arch,'revision',0))
    registry=registry or default_registry
    try:arch=ArchitectureIR.model_validate(arch.model_dump() if hasattr(arch,'model_dump') else arch)
    except Exception as exc:
        report.schema_valid=False;report.structural_valid=False;report.constructor_valid=False
        report.issues.append(ValidationIssue(category='schema',message=str(exc)));return report,None,None,None
    nodes={n.id:n for n in arch.nodes}
    # Only explicit Sequential data order and proven shape-preserving leaf adapters.
    transparent={'torch.nn.ReLU','torch.nn.GELU','torch.nn.SiLU','torch.nn.Dropout','torch.nn.Identity'}
    shape_checked=False
    for parent in arch.nodes:
        if parent.module_type!='torch.nn.Sequential':continue
        previous=None
        for registration in parent.children:
            node=nodes[registration.node_id]
            if node.module_type=='torch.nn.Linear':
                if previous is not None:
                    shape_checked=True
                    if previous.parameters.get('out_features')!=node.parameters.get('in_features'):
                        report.issues.append(ValidationIssue(category='shape',node=node.id,message=f"Sequential mismatch: {previous.qualified_name} outputs {previous.parameters.get('out_features')}, but {node.qualified_name} expects {node.parameters.get('in_features')}",suggested_fix='Submit an atomic batch updating both explicitly connected Linear dimensions.'))
                previous=node
            elif node.module_type not in transparent:previous=None
    report.shape_valid=False if any(i.category=='shape' for i in report.issues) else True if shape_checked else None
    try:model,weights,paths=build_architecture(arch,originals=originals,registry=registry)
    except Exception as exc:
        report.constructor_valid=False
        report.issues.append(ValidationIssue(category='constructor',node=getattr(exc,'node',None),message=str(exc),exception_type=type(exc).__name__))
        return report,None,None,None
    if args is None:return report,model,weights,paths
    hooks=[];stack=[]
    byobject={id(module):paths.get(path) for path,module in model.named_modules()}
    def pre(module,args,kwargs):stack.append(byobject.get(id(module)))
    def post(module,args,kwargs,out):stack.pop()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        # Validate a copy: BatchNorm, inplace inputs and backward must not mutate builds.
        trial=deepcopy(model)
        byobject={id(module):paths.get(path) for path,module in trial.named_modules()}
        for module in trial.modules():
            hooks.append(module.register_forward_pre_hook(pre,with_kwargs=True))
            hooks.append(module.register_forward_hook(post,with_kwargs=True))
        va,vk=deepcopy((args,kwargs or {}))
        with torch.random.fork_rng(devices=devices):
            with torch.set_grad_enabled(backward):out=trial(*va,**vk)
            report.forward_valid=True
            if report.shape_valid is not False:report.shape_valid=True
            if backward:
                values=[t for t in tensors(out) if t.requires_grad and (t.is_floating_point() or t.is_complex())]
                if not values:raise ValueError('No differentiable tensor outputs; supply a model with a differentiable output')
                sum(t.real.sum() for t in values).backward()
                report.backward_valid=True
    except Exception as exc:
        category='backward' if report.forward_valid else 'forward'
        if category=='forward':
            report.forward_valid=False
            if any(term in str(exc).lower() for term in ('shape','mat1','channel','size mismatch','normalized_shape')):report.shape_valid=False
        else:report.backward_valid=False
        report.issues.append(ValidationIssue(category=category,node=stack[-1] if stack else None,message=str(exc),exception_type=type(exc).__name__,traceback_summary=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__,limit=4))))
    finally:
        for h in hooks:h.remove()
    return report,model,weights,paths
