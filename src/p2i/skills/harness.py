"""Optional capability layer; existing Harness remains usable without a registry."""
import json
import sys
from copy import deepcopy
from pathlib import Path
import torch
from .schema import EvaluationRecord, EnvironmentInfo, HarnessFeedback

class SkillHarnessMixin:
    @property
    def skill_registry(self):
        if self._skill_registry is None:
            from .registry import SkillRegistry
            self._skill_registry=SkillRegistry()
        return self._skill_registry
    def skills(self,**filters):return self.skill_registry.list(**filters)
    def search_skills(self,**query):
        results=self.skill_registry.search(**query)
        self._event('skill_search',query=query,results=[r.model_dump(mode='json') for r in results]);return results
    def inspect_skill(self,skill_id,revision=None):return self.skill_registry.get(skill_id,revision)
    def validate_skill(self,skill_id,**kwargs):
        report=self.skill_registry.validate(skill_id,**kwargs)
        self._event('skill_validation',skill_id=skill_id,report=report.model_dump(mode='json'));return report
    def apply_skill(self,action):return self.apply(action)
    def _event(self,event,**details):
        from .schema import now
        self._experiments.append({'event':event,'revision':self._arch.revision,'timestamp':now().isoformat(),**details})
    def experiment_history(self):return deepcopy(self._experiments)
    def export_experiment(self,path):
        Path(path).write_text(json.dumps({'version':'0.1','model':self.observe(),'architecture':self.architecture().model_dump(mode='json'),'events':self.experiment_history()},sort_keys=True,indent=2)+'\n')
    def evaluate(self,evaluator=None):
        with self._lock:
            report=self.validate();model=self.build()
            metrics={'parameters':sum(p.numel() for p in model.parameters()),'modules':sum(1 for _ in model.modules()),'forward_success':int(report.forward_valid is True)}
            if self.observed_revision==self._arch.revision and self.latest_ir:metrics['operations']=len(self.latest_ir.operations)
            if evaluator is not None:
                if not report.valid:raise ValueError('Evaluator requires valid model')
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self._arch.seed);metrics.update(evaluator(model,deepcopy(self._args),deepcopy(self._kwargs)))
            record=EvaluationRecord(model_revision=self._arch.revision,skill_ids=sorted({n.skill_id for n in self._arch.nodes if n.skill_id}),metrics=metrics,environment=EnvironmentInfo(python=sys.version.split()[0],torch=torch.__version__,seed=self._arch.seed))
            self.last_evaluation=record;self._event('evaluation_outcome',evaluation=record.model_dump(mode='json'));return record
    def snapshot(self):
        from p2i.harness.core import Harness
        other=Harness.from_architecture(self.architecture(),example_inputs=deepcopy(self._args),example_kwargs=deepcopy(self._kwargs),registry=self.registry.copy(),skill_registry=self._skill_registry)
        other._originals=deepcopy(self._originals);other.latest_ir=deepcopy(self.latest_ir);other.observed_revision=self.observed_revision
        other._observed_bindings=deepcopy(self._observed_bindings);other._source_bindings=deepcopy(self._source_bindings);other.observed_configuration=deepcopy(self.observed_configuration)
        other._experiments=deepcopy(self._experiments);other._event('snapshot',source_revision=self._arch.revision)
        return other
    def feedback(self,action_result=None):
        return HarnessFeedback.model_validate({'revision':self._arch.revision,'action_result':action_result.model_dump(mode='json') if action_result else None,'validation':self.last_validation.model_dump(mode='json') if self.last_validation else None,'evaluation':self.last_evaluation.model_dump(mode='json') if self.last_evaluation else None,'skill_events':[e for e in self._experiments[-20:] if e['event'].startswith('skill_')],'warnings':[],'suggested_next_tools':['validate_model'] if not self.last_validation else ['retrace_model'] if self.observed_revision!=self._arch.revision else ['evaluate_model']})

def compare_evaluations(baseline,candidate):
    output={}
    for key in sorted(baseline.metrics.keys() & candidate.metrics.keys()):
        before,after=baseline.metrics[key],candidate.metrics[key]
        output[key]={'before':before,'after':after,'delta':after-before if type(before) in (int,float) and type(after) in (int,float) else None}
    return output
