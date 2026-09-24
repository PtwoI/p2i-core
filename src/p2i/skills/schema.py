"""Portable capability metadata. No source is executed by schema loading."""
from datetime import datetime, timezone
from typing import Annotated, Literal
from pydantic import Field, JsonValue, model_validator
from p2i.ir.model import Record
from p2i.architecture.schema import ArchitectureIR, TensorContract, ValidationIssue

def now(): return datetime.now(timezone.utc)
SkillKind = Literal['primitive','canonical','composite','custom','discovered']
SkillStatus = Literal['candidate','registered','compiled','validated','evaluated','promoted','deprecated','rejected']

class HyperparameterSpec(Record):
    name: str
    type: Literal['int','float','bool','str','array','any'] = 'any'
    default: JsonValue = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: list[JsonValue] = Field(default_factory=list)

class SkillConstraint(Record):
    kind: Literal['divisible','equal','device']
    left: str
    right: int | str

class RegisteredModuleImplementation(Record):
    type: Literal['registered_module'] = 'registered_module'
    module_type: str
class CompositeIRImplementation(Record):
    type: Literal['composite_ir'] = 'composite_ir'
    architecture: ArchitectureIR
class PythonModuleImplementation(Record):
    type: Literal['python_module'] = 'python_module'
    class_name: str
    source: str = Field(max_length=100_000)
    sha256: str = ''
Implementation = Annotated[RegisteredModuleImplementation | CompositeIRImplementation | PythonModuleImplementation, Field(discriminator='type')]

class SkillOrigin(Record):
    kind: Literal['builtin','user','model-extracted','external-agent','derived'] = 'user'
    source_model_id: str | None = None
    source_revision: int | None = None
    source_nodes: list[str] = Field(default_factory=list)
    source_file: str | None = None
    parent_skill_id: str | None = None
    parent_skill_revision: int | None = None
    reason: str | None = None

class EnvironmentInfo(Record):
    python: str
    torch: str
    device: str = 'cpu'
    dtype: str = 'float32'
    seed: int = 0
class EvaluationRecord(Record):
    model_revision: int
    skill_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str]
    timestamp: datetime = Field(default_factory=now)
    environment: EnvironmentInfo
class SkillValidationReport(Record):
    static_valid: bool = False
    import_valid: bool = False
    instantiate_valid: bool = False
    contract_valid: bool = False
    forward_valid: bool = False
    finite_valid: bool = False
    backward_valid: bool | None = None
    gradient_valid: bool | None = None
    retrace_valid: bool | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    environment: EnvironmentInfo | None = None
    cases: list[dict[str, JsonValue]] = Field(default_factory=list)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    fingerprint: str = ''
    @property
    def valid(self):
        return all((self.static_valid,self.import_valid,self.instantiate_valid,self.contract_valid,self.forward_valid,self.finite_valid,self.backward_valid,self.gradient_valid,self.retrace_valid)) and not self.issues
class SkillSpec(Record):
    version: Literal['0.1'] = '0.1'
    id: str = Field(pattern=r'^[a-z][a-z0-9_.-]{1,99}$')
    name: str
    kind: SkillKind
    status: SkillStatus = 'candidate'
    description: str
    semantic_tags: list[str] = Field(default_factory=list)
    input_contracts: list[TensorContract] = Field(default_factory=list)
    output_contracts: list[TensorContract] = Field(default_factory=list)
    hyperparameters: list[HyperparameterSpec] = Field(default_factory=list)
    constraints: list[SkillConstraint] = Field(default_factory=list)
    implementation: Implementation
    origin: SkillOrigin = Field(default_factory=SkillOrigin)
    validation: SkillValidationReport | None = None
    evaluation: list[EvaluationRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    revision: int = Field(default=0,ge=0)
class SkillSearchResult(Record):
    skill_id: str
    revision: int
    name: str
    score: float
    text_score: float
    contract_score: float
    status_score: float
    compatible: bool
    reasons: list[str]
class MissingCapability(Record):
    type: Literal['missing_capability'] = 'missing_capability'
    description: str
    required_inputs: list[TensorContract]
    required_outputs: list[TensorContract]
    constraints: list[str] = Field(default_factory=list)
class SkillError(ValueError):
    def __init__(self,code,message,details=None):
        super().__init__(message);self.code=code;self.details=details or {}

class HarnessFeedback(Record):
    revision: int
    action_result: dict[str, JsonValue] | None = None
    validation: dict[str, JsonValue] | None = None
    evaluation: EvaluationRecord | None = None
    skill_events: list[dict[str, JsonValue]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggested_next_tools: list[str] = Field(default_factory=list)
