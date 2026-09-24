import json
from copy import deepcopy
import pytest
import torch
from torch import nn
import p2i
from p2i.actions import SetParameter
from p2i.architecture import ArchitectureIR,ModuleAdapter,ModuleAdapterRegistry
from p2i.architecture.registry import registry
from p2i.architecture.builder import build_architecture
from p2i.examples import make_demo

@pytest.mark.parametrize('module',[
 nn.Linear(4,3),nn.Conv1d(2,4,3),nn.Conv2d(2,4,3),nn.ReLU(),nn.GELU(),nn.SiLU(),nn.Dropout(.1),nn.LayerNorm(4),nn.BatchNorm1d(4),nn.BatchNorm2d(4),nn.Embedding(10,4),nn.MultiheadAttention(8,2,batch_first=True),nn.Sequential(nn.Linear(4,2)),nn.ModuleList([nn.ReLU()])])
def test_adapters_roundtrip(module):
 adapter=registry.find(module);spec=adapter.extract(module);rebuilt=adapter.build(spec)
 assert adapter.extract(rebuilt)==spec

def test_unknown_and_nested():
 model,args=make_demo('transformer');h=p2i.Harness(model,example_inputs=args)
 assert not h.observe(node_id='')['editable']
 assert h.observe(node_id='blocks.0.mlp.0')['editable']
 assert h.observe(node_id='blocks')['kind']=='container'
 assert not h.apply({'type':'set_parameter','target':'','parameter':'width','value':20}).success

def test_atomic_hidden_dimension_edit_and_state():
 model,args=make_demo('transformer');h=p2i.Harness(model,example_inputs=args)
 original=h.retrace(static=False)
 before=h.architecture().model_dump()
 bad=h.apply(SetParameter(target='blocks.0.mlp.0',parameter='out_features',value=96))
 assert not bad.success and h.architecture().model_dump()==before
 action={'type':'batch','expected_revision':0,'actions':[{'type':'set_parameter','target':'blocks.0.mlp.0','parameter':'out_features','value':96},{'type':'set_parameter','target':'blocks.0.mlp.2','parameter':'in_features','value':96}]}
 preview=h.preview(action);assert preview.success and not preview.committed and h.observe()['revision']==0
 result=h.apply(action);assert result.success and result.validation.forward_valid
 assert h.observe()['revision']==1
 built=h.build();assert built.blocks[0].mlp[0].out_features==96
 assert torch.equal(model.head.weight,built.head.weight)
 assert torch.equal(model.blocks[1].mlp[0].weight,built.blocks[1].mlp[0].weight)
 assert h.last_weights.reinitialized and h.last_weights.dropped
 assert torch.equal(built.blocks[0].mlp[0].weight,h.build().blocks[0].mlp[0].weight)
 report=h.validate(backward=True);assert report.forward_valid and report.backward_valid
 changed=h.retrace(static=False)
 assert changed.modules[0].parameter_count<original.modules[0].parameter_count
 assert h.last_comparison['shape_changes']
 assert h.observe()['observed_revision']==1
 assert next(t.shape for t in changed.tensors if t.id==next(m for m in changed.modules if m.qualified_name=='blocks.0.mlp.0').output_tensor_ids[0])==[2,16,96]
 assert h.undo()['success'];assert h.observe()['revision']==2
 assert h.build().blocks[0].mlp[0].out_features==128
 assert h.redo()['success'];assert h.build().blocks[0].mlp[0].out_features==96
 assert len(h.diff(0,1).changes)==2
 assert not h.apply(action).success
 assert h.observe()['revision']==3

@pytest.mark.parametrize('value',[-1,0,2.5,True,'8'])
def test_invalid_linear(value):
 h=p2i.Harness(nn.Linear(4,2),example_inputs=(torch.ones(1,4),))
 assert not h.apply(SetParameter(target='a0',parameter='out_features',value=value)).success
 assert h.observe()['revision']==0

def test_attention_validation_and_owned_children():
 h=p2i.Harness(nn.MultiheadAttention(8,2,batch_first=True),example_inputs=(torch.randn(1,3,8),)*3)
 bad=h.apply(SetParameter(target='a0',parameter='num_heads',value=7))
 assert not bad.success and 'divisible' in str(bad.errors)
 assert not h.apply(SetParameter(target='out_proj',parameter='out_features',value=9)).success
 assert h.apply(SetParameter(target='a0',parameter='dropout',value=.2)).success

def test_structural_actions():
 model=nn.Sequential(nn.Linear(4,4),nn.ReLU(),nn.Linear(4,2)).eval();h=p2i.Harness(model,example_inputs=(torch.ones(1,4),))
 assert h.apply({'type':'replace_module','target':'1','replacement':{'module_type':'torch.nn.GELU'}}).success
 assert isinstance(h.build()[1],nn.GELU)
 assert h.apply({'type':'insert_module','parent':'a0','index':2,'module':{'module_type':'torch.nn.Dropout','parameters':{'p':.2}}}).success
 assert len(h.build())==4
 new=next(n for n in h.architecture().nodes if n.module_type=='torch.nn.Dropout')
 assert h.apply({'type':'remove_module','target':new.id}).success
 assert len(h.build())==3
 assert h.apply({'type':'wrap_module','target':'1','module':{'module_type':'torch.nn.Identity'}}).success
 assert h.validate().forward_valid
 assert h.undo()['success']
 assert not h.apply({'type':'remove_module','target':'a0'}).success

@pytest.mark.parametrize('name',['mlp','cnn','transformer'])
def test_rebuild_demos(name):
 model,args=make_demo(name);h=p2i.Harness(model,example_inputs=args)
 built=h.build()
 assert torch.allclose(model(*args),built(*args))
 if name=='cnn':
  bad=h.apply(SetParameter(target='features.3',parameter='out_channels',value=24))
  assert not bad.success and bad.validation.forward_valid is False
  assert h.apply(SetParameter(target='features.0',parameter='padding_mode',value='reflect')).success
 elif name=='mlp':
  assert h.apply({'type':'batch','actions':[{'type':'set_parameter','target':'layers.0','parameter':'out_features','value':24},{'type':'set_parameter','target':'layers.2','parameter':'in_features','value':24}]}).success
 else:assert h.apply(SetParameter(target='norm',parameter='eps',value=.001)).success
 assert h.retrace(static=False).modules

def test_schema_constructor_and_standalone(tmp_path):
 h=p2i.Harness(nn.Sequential(nn.Linear(4,3),nn.ReLU()),example_inputs=(torch.ones(2,4),))
 a=h.architecture();a.save(tmp_path/'arch.json');b=ArchitectureIR.load(tmp_path/'arch.json');assert a==b
 assert p2i.build(b)(torch.ones(2,4)).shape==(2,3)
 data=a.model_dump();data['nodes'][0]['children'].append({'name':'cycle','node_id':'a0'})
 with pytest.raises(ValueError,match='cycle'):ArchitectureIR.model_validate(data)
 bad=h.apply({'type':'replace_module','target':'1','replacement':{'module_type':'torch.nn.Linear','parameters':{}}})
 assert not bad.success and h.observe()['revision']==0
 assert not h.apply({'type':'insert_module','parent':'1','index':0,'module':{'module_type':'torch.nn.Identity'}}).success
 assert not h.apply({'type':'set_parameter','target':'unknown','parameter':'p','value':.3}).success
 assert not h.apply({'type':'set_parameter','target':'1','parameter':'x','value':1,'extra':'bad'}).success

def test_custom_registry():
 class Scale(nn.Module):
  def __init__(self,factor=2):super().__init__();self.factor=factor
  def forward(self,x):return x*self.factor
 class ScaleAdapter(ModuleAdapter):
  module_class=Scale
  def extract(self,m):return {'factor':m.factor}
  def validate(self,p):
   if type(p['factor']) not in (float,int):raise ValueError('factor must be numeric')
  def build(self,p):return Scale(**p)
 custom=registry.copy();custom.register('example.Scale',ScaleAdapter())
 h=p2i.Harness(Scale(),example_inputs=(torch.ones(2),),registry=custom)
 assert h.apply(SetParameter(target='a0',parameter='factor',value=3)).success
 assert torch.equal(h.build()(torch.ones(2)),torch.ones(2)*3)
 assert torch.equal(p2i.build(h.architecture(),registry=custom)(torch.ones(2)),torch.ones(2)*3)

def test_shared_modules_and_tied_weights():
 layer=nn.Linear(4,4);model=nn.Sequential(layer,layer)
 h=p2i.Harness(model,example_inputs=(torch.ones(1,4),))
 rebuilt=h.build();assert rebuilt[0] is rebuilt[1]
 left,right=nn.Linear(4,4),nn.Linear(4,4);right.weight=left.weight
 h=p2i.Harness(nn.Sequential(left,right),example_inputs=(torch.ones(1,4),))
 rebuilt=h.build();assert rebuilt[0].weight is rebuilt[1].weight

def test_capture_limits_and_observed_schema():
 from p2i.analyze.inspection import RuntimeInspection
 model=nn.Linear(4,2);x=torch.ones(1,4)
 capture=RuntimeInspection(max_tensors=2,max_elements=4,sample_size=2)
 ir=p2i.trace(model,(x,),static=False,inspection=capture)
 assert len(capture.tensors)<=2
 assert all(v['values'] is None or len(v['values'])<=2 for v in capture.tensors.values())
 assert ir.version=='0.1' and 'values' not in ir.tensors[0].model_dump()
 assert any(v['values'] is not None for v in capture.tensors.values())
 assert len(ir.operations)==len(p2i.trace(model,(x,),static=False).operations)
 with pytest.raises(ValueError):RuntimeInspection(max_elements=1000000)

def test_validation_preserves_rng_model_and_inputs():
 model=nn.Sequential(nn.BatchNorm1d(4),nn.Dropout(.3));x=torch.randn(3,4);rng=torch.get_rng_state().clone()
 h=p2i.Harness(model,example_inputs=(x,));before=deepcopy(model.state_dict())
 assert h.validate(backward=True).backward_valid
 assert torch.equal(rng,torch.get_rng_state())
 assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())

def test_optional_none_registration_preserved():
 class Optional(nn.Module):
  def __init__(self):super().__init__();self.add_module('optional',None);self.proj=nn.Linear(2,2)
  def forward(self,x):return self.proj(x) if self.optional is None else self.optional(x)
 h=p2i.Harness(Optional(),example_inputs=(torch.ones(1,2),))
 assert h.validate().forward_valid
 assert h.apply(SetParameter(target='proj',parameter='bias',value=False)).success

def test_modulelist_insertion_preserves_unaffected_state():
 class Loop(nn.Module):
  def __init__(self):super().__init__();self.blocks=nn.ModuleList([nn.Linear(2,2),nn.Linear(2,2)])
  def forward(self,x):
   for layer in self.blocks:x=layer(x)
   return x
 model=Loop();h=p2i.Harness(model,example_inputs=(torch.ones(1,2),))
 assert h.apply({'type':'insert_module','parent':'blocks','index':0,'module':{'module_type':'torch.nn.Identity'}}).success
 built=h.build();assert torch.equal(built.blocks[1].weight,model.blocks[0].weight)
 assert torch.equal(built.blocks[2].weight,model.blocks[1].weight)
 assert h.retrace(static=False).modules

def test_mha_state_and_readonly_artifact():
 module=nn.MultiheadAttention(8,2,bias=False,batch_first=True).eval()
 h=p2i.Harness(module,example_inputs=(torch.randn(1,3,8),)*3)
 built=h.build()
 assert all(torch.equal(value,built.state_dict()[key]) for key,value in module.state_dict().items())
 model,args=make_demo('mlp');h=p2i.Harness(model,example_inputs=args)
 loaded=p2i.Harness.from_architecture(h.architecture())
 report=loaded.validate();assert not report.constructor_valid and 'live Harness' in report.issues[0].message

def test_failed_retrace_retains_evidence_and_reports_forward_node():
 class Wrong(nn.Module):
  def __init__(self):super().__init__();self.layer=nn.Linear(2,2)
  def forward(self,x):return self.layer(x)
 h=p2i.Harness(Wrong(),example_inputs=(torch.ones(1,2),));original=h.retrace(static=False)
 h._args=(torch.ones(1,3),)
 report=h.validate();assert not report.forward_valid and report.issues[-1].node=='a1'
 with pytest.raises(ValueError):h.retrace()
 assert h.latest_ir==original and h.observed_revision==0

def test_observed_configuration_and_contract_provenance():
 model,args=make_demo('mlp');h=p2i.Harness(model,example_inputs=args)
 h.retrace(static=False)
 node=h.observe(node_id='layers.0')
 assert node['output_contracts'][0]['shape']==[2,32]
 assert node['source_observed_module_id']==node['latest_observed_module_id']
 assert h.provenance()
 h.apply(SetParameter(target='layers.0',parameter='bias',value=False))
 mid=node['latest_observed_module_id']
 assert h.observed_configuration[mid]['parameters']['bias'] is True
 h.retrace(static=False)
 assert h.observed_configuration[mid]['parameters']['bias'] is False
 assert h.observe(node_id='layers.0')['contracts_revision']==1

def test_artifact_cannot_edit_ignored_adapter_internal():
 h=p2i.Harness(nn.MultiheadAttention(8,2,batch_first=True))
 arch=h.architecture()
 child=next(n for n in arch.nodes if n.managed_by)
 child.managed_by=None;child.editable=True;child.module_type='torch.nn.Linear';child.parameters={'in_features':8,'out_features':8,'bias':True}
 loaded=p2i.Harness.from_architecture(arch)
 result=loaded.apply(SetParameter(target=child.id,parameter='out_features',value=9))
 assert not result.success and 'Adapter-owned child' in str(result.errors)
