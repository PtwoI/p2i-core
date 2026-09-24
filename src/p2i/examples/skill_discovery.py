"""Offline scripted controller: failure -> correction -> promotion -> reuse."""
from pathlib import Path
import json
import tempfile
import torch
from torch import nn
import p2i
from p2i.skills import (SkillSpec,SkillRegistry,SkillOrigin,PythonModuleImplementation,HyperparameterSpec,TensorContract,MissingCapability)

SOURCE='''import torch
from torch import nn
class GatedResidual(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return x + torch.sigmoid(self.gate) * self.projection(x)
'''
BAD_SOURCE=SOURCE.replace('nn.Linear(dim, dim)','nn.Linear(dim + 1, dim)')

def candidate(source=SOURCE):
    contract=TensorContract(shape=['B','T','dim'],dtype_family='floating',device='cpu',evidence='declared')
    return SkillSpec(id='discovered.gated_residual',name='GatedResidual',kind='discovered',description='Explicitly supplied differentiable gated sequence mixer',semantic_tags=['mixer','gated','sequence'],input_contracts=[contract],output_contracts=[contract],hyperparameters=[HyperparameterSpec(name='dim',type='int',default=8,minimum=1)],implementation=PythonModuleImplementation(class_name='GatedResidual',source=source),origin=SkillOrigin(kind='external-agent'))

def run(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    registry=SkillRegistry(directory/'skills.sqlite3',allow_python=True)
    torch.manual_seed(0)
    h=p2i.Harness(nn.Sequential(nn.Linear(8,8),nn.Identity()),example_inputs=(torch.randn(2,7,8),),skill_registry=registry)
    assert not h.search_skills(query='gated')
    missing=MissingCapability(description='Gated sequence mixer',required_inputs=candidate().input_contracts,required_outputs=candidate().output_contracts,constraints=['differentiable','preserve_sequence_length'])
    (directory/'missing.json').write_text(missing.model_dump_json(indent=2))
    first=registry.register(candidate(BAD_SOURCE));failure=h.validate_skill(first.id)
    assert not failure.valid and any(i.category=='forward' for i in failure.issues)
    (directory/'failure.json').write_text(failure.model_dump_json(indent=2))
    corrected=registry.update(candidate(),reason='External controller supplies corrected input dimension')
    assert corrected.revision>first.revision
    report=h.validate_skill(corrected.id);assert report.valid,report.model_dump_json()
    promoted=registry.promote(corrected.id)
    original=h.retrace(static=False);baseline=h.evaluate()
    action=p2i.ReplaceWithSkill(target='1',skill_id=promoted.id)
    assert h.preview(action).success
    result=h.apply(action);assert result.success,result.errors
    assert h.validate(backward=True).backward_valid
    updated=h.retrace(static=False);evaluation=h.evaluate()
    registry.record_evaluation(promoted.id,evaluation)
    h.export_experiment(directory/'experiment-a.json')
    # Different architecture, fresh learned state, same registered source.
    second=p2i.Harness(nn.Sequential(nn.Identity(),nn.Linear(8,3)),example_inputs=(torch.randn(3,11,8),),skill_registry=registry)
    second.retrace(static=False)
    assert second.search_skills(query='gated')[0].skill_id==promoted.id
    reused=second.apply(p2i.ReplaceWithSkill(target='0',skill_id=promoted.id));assert reused.success,reused.errors
    assert second.validate(backward=True).backward_valid
    second.retrace(static=False);second.evaluate();second.export_experiment(directory/'experiment-b.json')
    registry.export(directory/'skills.json')
    summary={'success':True,'skill_id':promoted.id,'skill_revision':promoted.revision,'model_a_revision':h.architecture().revision,'model_b_revision':second.architecture().revision,'comparison':p2i.compare_evaluations(baseline,evaluation),'before_modules':len(original.modules),'after_modules':len(updated.modules)}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=None);args=parser.parse_args()
    if args.output:print(json.dumps(run(args.output),indent=2))
    else:
        with tempfile.TemporaryDirectory() as path:print(json.dumps(run(path),indent=2))
