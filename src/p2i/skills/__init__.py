from .schema import (SkillSpec,SkillOrigin,HyperparameterSpec,SkillConstraint,RegisteredModuleImplementation,CompositeIRImplementation,PythonModuleImplementation,MissingCapability,SkillValidationReport,SkillSearchResult,EvaluationRecord,SkillError)
from .registry import SkillRegistry
from .extraction import skill_from_subgraph,extract_skill
from .harness import compare_evaluations
from p2i.architecture.schema import TensorContract

def ingest_skill(*,source,metadata,registry):
    """Store a statically checked candidate without importing its source."""
    spec=metadata.model_copy(deep=True) if isinstance(metadata,SkillSpec) else SkillSpec.model_validate(metadata)
    if spec.implementation.type!='python_module':raise SkillError('SCHEMA_ERROR','Ingestion requires Python module implementation')
    spec.implementation.source=source
    return registry.update(spec,reason='external code revision') if registry.versions(spec.id) else registry.register(spec)

def candidate_from_missing(missing,*,skill_id,name,source,class_name,hyperparameters=None):
    """External controller supplies code; P2I only transfers explicit contracts."""
    missing=missing if isinstance(missing,MissingCapability) else MissingCapability.model_validate(missing)
    return SkillSpec(id=skill_id,name=name,kind='discovered',description=missing.description,input_contracts=missing.required_inputs,output_contracts=missing.required_outputs,hyperparameters=hyperparameters or [],implementation=PythonModuleImplementation(class_name=class_name,source=source),origin=SkillOrigin(kind='external-agent',reason='Missing capability: '+', '.join(missing.constraints)))
