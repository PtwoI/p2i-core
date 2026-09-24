from dataclasses import dataclass
import pytest
import torch
from torch import nn
import p2i
from p2i.analyze.tree import tensors

@dataclass
class Result:
    hidden: torch.Tensor
    note: str

class Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.used=nn.Linear(4,4)
        self.shared=self.used
        self.unused=nn.Linear(4,4)
    def forward(self, x, *, mask):
        a=self.used(x)
        b=self.shared(a)*mask
        return {'tuple':(b, [a, 9, None]), 'result':Result(b,'ok')}

def test_nested_and_shared():
    ir=p2i.trace(Nested(), example_args=(torch.randn(2,4),),example_kwargs={'mask':torch.ones(2,4)},static=False)
    used=next(m for m in ir.modules if m.name=='used')
    assert used.aliases==['shared']
    assert len(used.call_ids)==2
    assert not next(m for m in ir.modules if m.name=='unused').executed
    assert len(ir.modules)==3
    assert len(ir.runtime.output_tensor_ids)==2
    assert all(t.shape==[2,4] for t in ir.tensors if t.id in ir.runtime.output_tensor_ids)
    assert sum(e.shared for e in ir.module_edges)==1
    assert next(m for m in ir.modules if m.id=='m0').origin.kind=='user'
    assert used.origin.kind=='framework'

def test_tree_cycles():
    t=torch.ones(1); values=[t];values.append(values)
    assert tensors(values)==[t]

def test_input_aliases_and_inplace_versions():
    class Inplace(nn.Module):
        def forward(self,x):
            y=x+1
            y.add_(2)
            return y*2
    x=torch.zeros(2)
    ir=p2i.trace(Inplace(),(x,),static=False)
    assert torch.equal(x,torch.zeros(2))
    op=next(o for o in ir.operations if 'add_' in o.op_type)
    assert set(op.input_tensor_ids).isdisjoint(op.output_tensor_ids)
    assert any(t.alias_of for t in ir.tensors)

def test_state_and_rng_unchanged():
    model=nn.Sequential(nn.BatchNorm1d(4),nn.Dropout())
    x=torch.randn(3,4)
    state={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.random.get_rng_state().clone()
    p2i.trace(model,(x,))
    assert torch.equal(rng,torch.random.get_rng_state())
    assert model.training
    assert all(torch.equal(state[k],v) for k,v in model.state_dict().items())
    assert not model._forward_hooks

@pytest.mark.parametrize('output',[None,42,'text'])
def test_scalar_outputs(output):
    class Scalar(nn.Module):
        def forward(self,x): return output
    ir=p2i.trace(Scalar(),(torch.ones(2),),static=False)
    assert ir.runtime.output_tensor_ids==[]
    assert ir.runtime.calls[0].completed

def test_invalid_api():
    with pytest.raises(TypeError,match='nn.Module'): p2i.trace(object())
    with pytest.raises(TypeError,match='tuple'): p2i.trace(nn.Identity(),torch.ones(1))
    with pytest.raises(ValueError,match='not both'): p2i.trace(nn.Identity(),(),example_args=())

def test_shared_container_aliases():
    class SharedContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=nn.Sequential(nn.Linear(4,4))
            self.b=self.a
        def forward(self,x): return self.b(self.a(x))
    ir=p2i.trace(SharedContainer(),(torch.ones(2,4),))
    assert len(ir.modules)==3
    child=next(m for m in ir.modules if m.qualified_name=='a.0')
    assert len(child.call_ids)==2
    assert all(o.parent_module_id for o in ir.operations if o.graph=='export')
