import json
import subprocess
import sys
import pytest
import torch
from torch import nn
import p2i
from p2i.skills import *
from p2i.examples.skill_discovery import candidate,SOURCE,BAD_SOURCE
from p2i.skills.contracts import compatible,parameters_for
from p2i.skills.application import compatible_replacements
from p2i.tools import dispatch,tool_schemas

@pytest.fixture
def registry(tmp_path):return SkillRegistry(tmp_path/'skills.db',allow_python=True)

def test_schema_contracts():
 c=TensorContract(shape=['B','T',8],dtype_family='floating')
 assert c.rank==3
 assert compatible(c,TensorContract(shape=[2,3,8],dtype='torch.float32'))
 assert not compatible(c,TensorContract(shape=[2,3,7]))
 assert not compatible(c,TensorContract(shape=[2,3,8],dtype='torch.int64'))
 with pytest.raises(ValueError):TensorContract(shape=[2,3],rank=3)
 s=candidate();assert SkillSpec.model_validate_json(s.model_dump_json())==s

@pytest.mark.parametrize('query,expected',[('dropout','builtin.dropout'),('attention','builtin.multihead_attention'),('linear','builtin.linear')])
def test_builtin_search(registry,query,expected):
 assert registry.search(query)[0].skill_id==expected
 assert len(registry.list())>=19
 assert registry.get(expected).status=='promoted'

def test_registry_candidate_lifecycle(registry,tmp_path):
 s=registry.register(candidate());assert s.status=='candidate'
 assert registry.register(candidate()).revision==0
 assert registry.search('gated')==[]
 assert registry.search('gated',status=['candidate'])
 with pytest.raises(SkillError,match='Promotion'):registry.promote(s.id)
 changed=candidate(BAD_SOURCE);v2=registry.update(changed,reason='test')
 assert v2.revision==1 and v2.origin.parent_skill_revision==0
 assert registry.get(s.id,0).implementation.source==SOURCE
 assert registry.reject(s.id).status=='rejected'
 assert registry.deprecate(s.id).status=='deprecated'
 file=tmp_path/'skills.json';registry.export(file)
 r2=SkillRegistry(tmp_path/'other.db');r2.import_file(file)
 assert r2.get(s.id).status=='candidate'
 assert r2.get(s.id).implementation.source==BAD_SOURCE

@pytest.mark.parametrize('source',[
 'broken syntax *',
 SOURCE.replace('import torch','import subprocess'),
 SOURCE.replace('return x +','return eval("1") +'),
 SOURCE.replace('return x +','return open("oops", "w") +'),
 SOURCE.replace('import torch','import requests'),
 SOURCE+'\nprint("side effect")',
 SOURCE.replace('nn.Module','object'),
])
def test_static_rejection(registry,source):
 with pytest.raises(SkillError):registry.register(candidate(source))
 assert not registry.versions('discovered.gated_residual')

@pytest.mark.parametrize('source,stage',[
 (BAD_SOURCE,'forward'),
 (SOURCE.replace('from torch import nn','from torch import missing_dependency_name\nfrom torch import nn'),'import'),
 (SOURCE.replace('self.projection = nn.Linear(dim, dim)','self.projection = nn.Linear(-1, dim)'),'instantiate'),
 (SOURCE.replace('return x + torch.sigmoid(self.gate) * self.projection(x)','return x[:, :, :2]'),'contract'),
 (SOURCE.replace('return x + torch.sigmoid(self.gate) * self.projection(x)','return x * float("nan")'),'finite'),
 (SOURCE.replace('return x + torch.sigmoid(self.gate) * self.projection(x)','return x.detach()'),'backward'),
 (SOURCE.replace('return x + torch.sigmoid(self.gate) * self.projection(x)','return torch.sqrt(x * 0)'),'gradient'),
])
def test_candidate_failures(registry,source,stage):
 s=registry.register(candidate(source));report=registry.validate(s.id)
 assert not report.valid and any(i.category==stage for i in report.issues),report
 assert registry.get(s.id).status=='candidate'
 with pytest.raises(SkillError):registry.promote(s.id)

@pytest.mark.parametrize('where',['import','instantiate','forward'])
def test_timeout(registry,where):
 if where=='import':
  # Default argument is evaluated during import, bounded by worker timeout.
  source=SOURCE.replace('dim=8','dim=sum(iter(int, 1))')
 elif where=='instantiate':source=SOURCE.replace('self.projection = nn.Linear(dim, dim)','while True: pass\n        self.projection = nn.Linear(dim, dim)')
 else:source=SOURCE.replace('return x + torch.sigmoid(self.gate) * self.projection(x)','while True: pass')
 s=registry.register(candidate(source));report=registry.validate(s.id,timeout=4)
 assert not report.valid
 assert report.issues[0].exception_type in ('TIMEOUT','WORKER_FAILED')

def test_search_contract_filter(registry):
 registry.register(candidate())
 assert not registry.search('gated',status=['candidate'],input_contract={'shape':[2,7,4]},parameters={'dim':8})
 assert registry.search('gated',status=['candidate'],input_contract={'shape':[2,7,8]},parameters={'dim':8})
 assert not registry.search('gated',status=['candidate'],tags=['not-present'])

def test_parameter_constraints(registry):
 s=registry.get('builtin.dropout')
 with pytest.raises(SkillError):parameters_for(s,{'p':2.0})
 with pytest.raises(SkillError):parameters_for(s,{'unknown':2})
 with pytest.raises(SkillError):parameters_for(registry.get('builtin.linear'),{})

def test_apply_builtin_and_rollback(registry):
 h=p2i.Harness(nn.Sequential(nn.Linear(8,8),nn.ReLU()),example_inputs=(torch.randn(2,8),),skill_registry=registry)
 h.retrace(static=False);original=h.build()[0].weight.detach().clone()
 action=p2i.ReplaceWithSkill(target='1',skill_id='builtin.gelu',expected_revision=0)
 assert h.preview(action).success and h.architecture().revision==0
 assert h.apply(action).success
 assert h.observe(node_id='1')['skill_id']=='builtin.gelu'
 assert torch.equal(h.build()[0].weight,original)
 assert not h.apply(action).success
 assert h.apply(p2i.InsertSkill(parent='',index=1,skill_id='builtin.dropout',parameters={'p':.2})).success
 assert h.validate(backward=True).backward_valid
 assert h.undo()['success']
 assert h.architecture().revision==3
 assert h.build()[1].__class__ is nn.GELU
 assert compatible_replacements(h,'1')

def test_incompatible_and_unknown(registry):
 h=p2i.Harness(nn.Sequential(nn.Linear(8,8)),example_inputs=(torch.randn(2,8),),skill_registry=registry)
 h.retrace(static=False)
 r=h.apply(p2i.ReplaceWithSkill(target='0',skill_id='builtin.linear',parameters={'in_features':9,'out_features':8}))
 assert not r.success and r.error_code=='CONTRACT_MISMATCH'
 assert h.architecture().revision==0
 assert not h.apply(p2i.ReplaceWithSkill(target='0',skill_id='missing.skill')).success

def test_composite_extraction(registry):
 h=p2i.Harness(nn.Sequential(nn.Linear(8,12),nn.GELU(),nn.Linear(12,8)),example_inputs=(torch.randn(2,8),),skill_registry=registry)
 c=TensorContract(shape=['B',8],dtype_family='floating')
 skill=p2i.skill_from_subgraph(h.architecture(),node_ids=['0','1','2'],name='ProjectionPair',skill_id='user.projection',input_contracts=[c],output_contracts=[c])
 registry.register(skill);assert registry.validate(skill.id).valid
 registry.promote(skill.id)
 other=p2i.Harness(nn.Sequential(nn.Identity()),example_inputs=(torch.randn(2,8),),skill_registry=registry)
 assert other.apply(p2i.ReplaceWithSkill(target='0',skill_id=skill.id)).success
 assert other.validate(backward=True).backward_valid
 assert len(other.retrace(static=False).operations)>0
 with pytest.raises(SkillError):p2i.skill_from_subgraph(h.architecture(),node_ids=['0','2'],name='Bad',skill_id='user.bad',input_contracts=[c],output_contracts=[c])

def test_tools_and_evaluation(registry,tmp_path):
 h=p2i.Harness(nn.Sequential(nn.Linear(8,8)),example_inputs=(torch.randn(2,8),),skill_registry=registry)
 assert len(tool_schemas())>=16
 assert dispatch({'tool':'inspect_model'},harness=h).success
 assert not dispatch({'tool':'inspect_node','arguments':{'node_id':'oops'}},harness=h).success
 assert dispatch({'tool':'search_skills','arguments':{'query':'dropout'}},harness=h).success
 assert dispatch({'tool':'nonsense'},harness=h).error.code=='UNKNOWN_TOOL'
 assert dispatch({'tool':'validate_skill','arguments':{'skill_id':'x','timeout':0}},harness=h).error.code=='SCHEMA_ERROR'
 a=h.evaluate();assert a.metrics['parameters']==72
 h.apply({'type':'set_parameter','target':'0','parameter':'out_features','value':4})
 b=h.evaluate(lambda model,args,kwargs:{'score':float(model(*args).mean().detach())})
 assert p2i.compare_evaluations(a,b)['parameters']['delta']==-36
 clone=h.snapshot();assert clone.architecture().revision==h.architecture().revision
 clone.apply({'type':'set_parameter','target':'0','parameter':'out_features','value':2})
 assert h.build()[0].out_features==4
 h.export_experiment(tmp_path/'history.json');assert json.loads((tmp_path/'history.json').read_text())['events']
 assert h.feedback().suggested_next_tools

def test_candidate_cannot_bypass_via_saved_architecture(registry):
 s=registry.register(candidate())
 h=p2i.Harness(nn.Identity())
 arch=h.architecture();arch.nodes[0].skill_id=s.id;arch.nodes[0].skill_revision=s.revision
 arch.nodes[0].module_type=f'p2i.skill.{s.id}.r{s.revision}'
 with pytest.raises(SkillError,match='validation'):p2i.Harness.from_architecture(arch,skill_registry=registry)

def test_no_global_torch_mutation(registry):
 source=SOURCE.replace('self.gate = nn.Parameter(torch.zeros(dim))','nn.Module.forward = self.forward\n        self.gate = nn.Parameter(torch.zeros(dim))')
 with pytest.raises(SkillError,match='Mutation'):registry.register(candidate(source))
