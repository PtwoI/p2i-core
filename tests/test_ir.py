import json
import pytest
import torch
import p2i
from p2i.ir import ModelIR
from p2i.examples import make_demo

@pytest.fixture
def ir():
    model, args = make_demo('mlp')
    return p2i.trace(model, args, static=False)

def test_roundtrip(ir, tmp_path):
    path = tmp_path / 'model.json'
    ir.save(path)
    assert p2i.load(path) == ir
    first = path.read_bytes()
    ir.save(path)
    assert path.read_bytes() == first
    assert 'data' not in json.loads(first)['tensors'][0]

def test_stable_ids():
    model, args = make_demo('mlp')
    a, b = p2i.trace(model, args), p2i.trace(model, args)
    for key in ('modules', 'operations', 'tensors'):
        assert [x.id for x in getattr(a,key)] == [x.id for x in getattr(b,key)]

def test_references_and_edges(ir):
    ModelIR.model_validate(ir.model_dump())
    assert ir.data_edges
    assert all(e.child_id in next(m.children for m in ir.modules if m.id==e.parent_id) for e in ir.module_edges)

@pytest.mark.parametrize('corruption', ['tensor','edge','id','version'])
def test_reject_invalid_ir(ir, corruption):
    data = ir.model_dump()
    if corruption == 'tensor': data['operations'][0]['input_tensor_ids']=['bogus']
    if corruption == 'edge': data['data_edges']=[]
    if corruption == 'id': data['modules'].append(data['modules'][0])
    if corruption == 'version': data['version']='99'
    with pytest.raises(ValueError): ModelIR.model_validate(data)

def test_symbolic_shapes():
    class Dynamic(torch.nn.Module):
        def forward(self,x): return x.sin()+1
    ir = p2i.trace(Dynamic(), (torch.randn(3,4),), dynamic_shapes={'x':{0:torch.export.Dim('batch',min=1,max=10)}})
    assert any(any(isinstance(d,str) for d in t.shape) for t in ir.tensors if t.graph=='export')
