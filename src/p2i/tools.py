"""Provider-neutral JSON tools. No agent SDK or model dependency."""
from typing import Literal
from pydantic import Field, JsonValue
from p2i.ir.model import Record
from p2i.architecture.schema import TensorContract
from p2i.skills.schema import SkillSpec, SkillError

class Empty(Record):pass
class InspectModel(Record):level: Literal['summary','architecture']='summary'
class InspectNode(Record):node_id: str
class InspectRevision(Record):
    from_revision: int | None = None
    to_revision: int | None = None
class ListSkills(Record):
    status: str | None = None
    kind: str | None = None
class SearchSkills(Record):
    query: str = ''
    input_contract: TensorContract | None = None
    output_contract: TensorContract | None = None
    tags: list[str] | None = None
    status: list[str] | None = None
    parameters: dict[str, JsonValue] | None = None
class SkillId(Record):skill_id: str
class InspectSkill(SkillId):revision: int | None = None
class ValidateSkill(SkillId):
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    dimensions: dict[str, int] = Field(default_factory=dict)
    timeout: float = Field(default=30,gt=0,le=300)
    seed: int = 0
class RegisterSkill(Record):skill: SkillSpec
class EditAction(Record):action: dict[str, JsonValue]
class ValidateModel(Record):backward: bool = False
class ToolRequest(Record):
    version: Literal['0.1']='0.1'
    tool: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
class ToolError(Record):
    code: str
    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)
class ToolResult(Record):
    success: bool
    result: JsonValue = None
    error: ToolError | None = None

SCHEMAS={
 'inspect_model':InspectModel,'inspect_node':InspectNode,'inspect_revision':InspectRevision,
 'list_skills':ListSkills,'search_skills':SearchSkills,'inspect_skill':InspectSkill,
 'preview_action':EditAction,'apply_action':EditAction,'validate_model':ValidateModel,
 'build_model':Empty,'retrace_model':Empty,'evaluate_model':Empty,
 'register_skill':RegisterSkill,'validate_skill':ValidateSkill,'promote_skill':SkillId,'reject_skill':SkillId,
}
def tool_schemas():return [{'name':name,'input_schema':cls.model_json_schema()} for name,cls in SCHEMAS.items()]

def dispatch(request,*,harness=None,registry=None):
    """Always return a typed result for expected validation and execution failures."""
    try:
        req=ToolRequest.model_validate(request)
        if req.tool not in SCHEMAS:raise SkillError('UNKNOWN_TOOL',f'Unknown tool: {req.tool}')
        args=SCHEMAS[req.tool].model_validate(req.arguments).model_dump(exclude_none=True)
        if registry is None:
            from p2i.skills import SkillRegistry
            registry=harness.skill_registry if harness else SkillRegistry()
        name=req.tool
        if name=='list_skills':out=registry.list(**args)
        elif name=='search_skills':out=harness.search_skills(**args) if harness else registry.search(**args)
        elif name=='inspect_skill':out=registry.get(**args)
        elif name=='register_skill':out=registry.register(args['skill'])
        elif name=='validate_skill':out=harness.validate_skill(**args) if harness else registry.validate(**args)
        elif name in ('promote_skill','reject_skill'):out=getattr(registry,name.split('_')[0])(**args)
        else:
            if harness is None:raise SkillError('HARNESS_REQUIRED','This tool requires a live Harness or architecture artifact')
            if name=='inspect_model':out=harness.observe(**args)
            elif name=='inspect_node':out=harness.observe(**args)
            elif name=='inspect_revision':out=harness.diff(**args)
            elif name in ('preview_action','apply_action'):out=getattr(harness,name.split('_')[0])(args['action'])
            elif name=='validate_model':out=harness.validate(**args)
            elif name=='build_model':
                model=harness.build();out={'parameters':sum(p.numel() for p in model.parameters()),'weight_transfer':harness.last_weights.model_dump(mode='json')}
            elif name=='retrace_model':
                ir=harness.retrace();out={'revision':harness.observed_revision,'modules':len(ir.modules),'operations':len(ir.operations)}
            elif name=='evaluate_model':out=harness.evaluate()
        success=getattr(out,'success',getattr(out,'valid',True))
        error_code=getattr(out,'error_code',None) or 'VALIDATION_FAILED'
        if hasattr(out,'model_dump'):out=out.model_dump(mode='json')
        elif isinstance(out,list):out=[x.model_dump(mode='json') if hasattr(x,'model_dump') else x for x in out]
        return ToolResult(success=bool(success),result=out,error=None if success else ToolError(code=error_code,message='Inspect structured result for failed validation gates'))
    except Exception as exc:
        from pydantic import ValidationError
        return ToolResult(success=False,error=ToolError(code=getattr(exc,'code','SCHEMA_ERROR' if isinstance(exc,ValidationError) else 'EXECUTION_FAILED'),message=str(exc),details=getattr(exc,'details',{})))
