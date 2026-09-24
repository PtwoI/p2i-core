"""Disposable CPU validator process. Request/results use JSON, never pickle."""
import json
import sys
import traceback
from pathlib import Path
import torch
from .schema import SkillSpec, SkillValidationReport, EnvironmentInfo
from .contracts import parameters_for, compatible
from .static import inspect_source
from p2i.architecture.schema import TensorContract, ValidationIssue
from p2i.analyze.tree import tensors as flatten_tensors


def load_class(implementation):
    inspect_source(implementation.source,implementation.class_name)
    namespace={'__name__':'p2i_candidate'}
    exec(compile(implementation.source,'skill.py','exec'),namespace)
    return namespace[implementation.class_name]

def validate(request, progress):
    spec=SkillSpec.model_validate(request['skill']);report=SkillValidationReport(static_valid=True,parameters=request['parameters'],fingerprint=request['fingerprint'])
    report.environment=EnvironmentInfo(python=sys.version.split()[0],torch=torch.__version__,seed=request['seed'])
    stage='import'
    def step(name):
        nonlocal stage
        stage=name;Path(progress).write_text(name)
    try:
        torch.set_num_threads(1);torch.manual_seed(request['seed'])
        step('import')
        if spec.implementation.type=='python_module':factory=load_class(spec.implementation)
        elif spec.implementation.type=='registered_module':
            from p2i.architecture.registry import registry
            factory=lambda **p:registry.get(spec.implementation.module_type).build(p)
        else:
            from p2i.architecture.builder import build
            factory=lambda **p:build(spec.implementation.architecture)
        report.import_valid=True
        step('instantiate');model=factory(**report.parameters);report.instantiate_valid=True
        if not isinstance(model,torch.nn.Module):raise TypeError('Implementation must build nn.Module')
        if not spec.input_contracts or not spec.output_contracts:raise ValueError('Explicit input and output contracts required for validation')
        for dimensions in request['cases']:
            step('contract');bindings={**report.parameters,**dimensions}
            inputs=[]
            for c in spec.input_contracts:
                if c.device not in (None,'cpu'):raise ValueError('Candidate validation requires CPU contracts')
                if not c.shape:raise ValueError('Concrete rank/shape is required for synthetic inputs')
                shape=[d if isinstance(d,int) else bindings.get(d) for d in c.shape]
                if any(type(d) is not int or d<=0 for d in shape):raise ValueError(f'Unbound or invalid input dimension: {c.shape}')
                import math
                if math.prod(shape)>1_000_000:raise ValueError('Synthetic input exceeds 1,000,000 elements')
                dtype=getattr(torch,(c.dtype or ('int64' if c.dtype_family=='integer' else 'float32')).removeprefix('torch.'))
                if dtype.is_floating_point:x=torch.randn(shape,dtype=dtype,requires_grad=True)
                else:x=torch.zeros(shape,dtype=dtype)
                inputs.append(x)
            step('forward');model.zero_grad(set_to_none=True);output=model(*inputs);tensors=list(flatten_tensors(output))
            # flatten_tensors yields tensors only in the Phase 1 tree utility.
            if not tensors:raise ValueError('Forward produced no tensor outputs')
            report.forward_valid=True
            step('contract')
            actual=[TensorContract(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device)) for t in tensors]
            if len(actual)!=len(spec.output_contracts) or any(not compatible(e,a,bindings) for e,a in zip(spec.output_contracts,actual)):raise ValueError('Output contract mismatch')
            report.contract_valid=True
            step('finite')
            if any(not torch.isfinite(t).all().item() for t in tensors):raise ValueError('Non-finite output')
            report.finite_valid=True
            step('backward');loss=sum(t.float().square().mean() for t in tensors if t.requires_grad)
            if not isinstance(loss,torch.Tensor):raise ValueError('No differentiable output; backward is unavailable')
            loss.backward();report.backward_valid=True
            step('gradient');grads=[x.grad for x in list(model.parameters())+inputs if x.requires_grad and x.grad is not None]
            if not grads or any(not torch.isfinite(g).all().item() for g in grads):raise ValueError('Missing or non-finite gradients')
            report.gradient_valid=True
            step('retrace')
            from p2i import trace
            ir=trace(model,example_inputs=tuple(x.detach() for x in inputs),static=False)
            if any(a.status!='success' for a in ir.metadata.analysis if a.technique in ('runtime_hooks','runtime_dispatch')):raise ValueError('Runtime re-trace failed')
            report.retrace_valid=True
            report.cases.append({'dimensions':dimensions,'inputs':[list(x.shape) for x in inputs],'outputs':[list(t.shape) for t in tensors],'modules':len(ir.modules),'operations':len(ir.operations)})
    except Exception as exc:
        field={'import':'import_valid','instantiate':'instantiate_valid','contract':'contract_valid','forward':'forward_valid','finite':'finite_valid','backward':'backward_valid','gradient':'gradient_valid','retrace':'retrace_valid'}[stage]
        setattr(report,field,False)
        report.issues.append(ValidationIssue(category=stage,message=str(exc),exception_type='DEPENDENCY_MISSING' if isinstance(exc,ImportError) else type(exc).__name__,traceback_summary=traceback.format_exc()[-3000:]))
    return report

def main():
    request=json.loads(Path(sys.argv[1]).read_text())
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU,(request['cpu_seconds'],request['cpu_seconds']+1))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024*1024,8*1024*1024))
        resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    except (ImportError,ValueError,OSError):pass
    report=validate(request,sys.argv[3]);Path(sys.argv[2]).write_text(report.model_dump_json())
if __name__=='__main__':main()
