"""Resolve skills into the existing constructor and transaction pipeline."""
from copy import deepcopy
from p2i.architecture.registry import ModuleAdapter
from p2i.architecture.schema import ConstructorSpec, TensorContract
from .registry import ELIGIBLE, fingerprint
from .contracts import check_contracts, parameters_for, compatible
from .schema import SkillError

class SkillAdapter(ModuleAdapter):
    owns_children=True
    def __init__(self,spec,registry):
        self.spec=spec;self.registry=registry;self.module_class=None
    def matches(self,module):return getattr(module,'_p2i_skill_key',None)==self.module_type
    def extract(self,module):return deepcopy(module._p2i_skill_parameters)
    def validate(self,parameters):
        values=parameters_for(self.spec,parameters)
        if self.spec.implementation.type=='python_module' and values!=self.spec.validation.parameters:
            raise SkillError('VALIDATION_REQUIRED','Python skill parameters differ from validated configuration')
    def build(self,parameters):
        self.validate(parameters)
        if self.spec.implementation.type=='composite_ir':
            from p2i.architecture.builder import build
            model=build(self.spec.implementation.architecture,registry=self.registry)
        else:
            from .worker import load_class
            model=load_class(self.spec.implementation)(**parameters)
        model._p2i_skill_key=self.module_type;model._p2i_skill_parameters=deepcopy(parameters)
        return model

def ensure_adapter(harness,spec):
    if spec.status not in ELIGIBLE:raise SkillError('VALIDATION_REQUIRED','Pinned skill revision has not passed validation')
    if spec.origin.kind!='builtin' and (not spec.validation or not spec.validation.valid or spec.validation.fingerprint!=fingerprint(spec)):
        raise SkillError('VALIDATION_REQUIRED','Skill requires a current successful validation report')
    if spec.implementation.type=='registered_module':return spec.implementation.module_type
    if spec.implementation.type=='python_module' and not harness.skill_registry.allow_python:
        raise SkillError('LOCAL_CODE_OPT_IN_REQUIRED','Applying Python skills executes locally. Construct SkillRegistry(..., allow_python=True) only for source you trust.')
    key=f'p2i.skill.{spec.id}.r{spec.revision}'
    if key not in harness.registry.adapters:harness.registry.register(key,SkillAdapter(spec,harness.registry))
    return key

def concrete_builtin_contracts(spec,node,parameters):
    """Known module adapter shape rules, never full-architecture assumptions."""
    if spec.implementation.type!='registered_module':return spec
    spec=spec.model_copy(deep=True)
    typ=spec.implementation.module_type
    same={'torch.nn.Dropout','torch.nn.ReLU','torch.nn.GELU','torch.nn.SiLU','torch.nn.Identity','torch.nn.Softmax'}
    if typ in same and len(node.input_contracts)==1:
        spec.input_contracts=deepcopy(node.input_contracts);spec.output_contracts=deepcopy(node.input_contracts)
    elif typ=='torch.nn.Linear' and len(node.input_contracts)==1 and node.input_contracts[0].shape:
        incoming=node.input_contracts[0].model_copy(deep=True);out=incoming.model_copy(deep=True)
        incoming.shape[-1]=parameters['in_features'];out.shape[-1]=parameters['out_features']
        spec.input_contracts=[incoming];spec.output_contracts=[out]
    return spec

def resolve_action(harness,arch,action):
    registry=harness.skill_registry
    spec=registry.get(action.skill_id,action.skill_revision)
    current=registry.get(action.skill_id)
    if spec.status not in ELIGIBLE or current.status in ('deprecated','rejected'):raise SkillError('SKILL_NOT_REUSABLE',f'Skill status {spec.status} is not eligible for reuse')
    parameters=parameters_for(spec,action.parameters)
    if action.type=='replace_with_skill':
        target=harness._resolve(arch,action.target)
        effective=concrete_builtin_contracts(spec,target,parameters)
        check_contracts(effective,target.input_contracts,target.output_contracts,parameters)
    else:
        parent=harness._resolve(arch,action.parent)
        nodes={n.id:n for n in arch.nodes}
        if parent.module_type=='torch.nn.Sequential' and 0<=action.index<=len(parent.children):
            incoming=nodes[parent.children[action.index-1].node_id].output_contracts if action.index else parent.input_contracts
            outgoing=nodes[parent.children[action.index].node_id].input_contracts if action.index<len(parent.children) else parent.output_contracts
            boundary=parent.model_copy(update={'input_contracts':incoming,'output_contracts':outgoing})
            effective=concrete_builtin_contracts(spec,boundary,parameters)
            check_contracts(effective,incoming,outgoing,parameters)
    # Custom constructor parameters must have been validated in a bounded worker before host use.
    if spec.implementation.type=='python_module' and spec.validation and parameters!=spec.validation.parameters:
        raise SkillError('VALIDATION_REQUIRED','Python skill parameters must match the validated configuration; validate this configuration first')
    module_type=ensure_adapter(harness,spec)
    return spec,ConstructorSpec(module_type=module_type,parameters=parameters)

def compatible_replacements(harness,target):
    node=harness._resolve(harness.architecture(),target);result=[]
    if not node.input_contracts or not node.output_contracts:return result
    for spec in harness.skill_registry.list(status=sorted(ELIGIBLE)):
        try:
            params=parameters_for(spec,{})
            effective=concrete_builtin_contracts(spec,node,params)
            if not effective.input_contracts or not effective.output_contracts:continue
            check_contracts(effective,node.input_contracts,node.output_contracts,params)
            if spec.implementation.type=='python_module' and not harness.skill_registry.allow_python:continue
            result.append({'skill':spec.model_dump(mode='json'),'parameters':params})
        except (ValueError,KeyError):continue
    return result
