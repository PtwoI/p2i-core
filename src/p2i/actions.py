"""Versioned JSON actions; no natural-language or Python execution boundary."""
from typing import Annotated, Literal, Union
from pydantic import Field, JsonValue, StrictInt, TypeAdapter
from p2i.ir.model import Record
from p2i.architecture.schema import ConstructorSpec, ArchitectureDiff, ValidationReport

class Action(Record):
    version: Literal['0.1']='0.1'
    expected_revision: StrictInt | None = None
class SetParameter(Action):
    type: Literal['set_parameter']='set_parameter'
    target: str
    parameter: str
    value: JsonValue
class ReplaceModule(Action):
    type: Literal['replace_module']='replace_module'
    target: str
    replacement: ConstructorSpec
class InsertModule(Action):
    type: Literal['insert_module']='insert_module'
    parent: str
    index: StrictInt
    module: ConstructorSpec
class RemoveModule(Action):
    type: Literal['remove_module']='remove_module'
    target: str
class WrapModule(Action):
    type: Literal['wrap_module']='wrap_module'
    target: str
    module: ConstructorSpec
    position: Literal['before','after']='after'
class ReplaceWithSkill(Action):
    type: Literal['replace_with_skill']='replace_with_skill'
    target: str
    skill_id: str
    skill_revision: StrictInt | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
class InsertSkill(Action):
    type: Literal['insert_skill']='insert_skill'
    parent: str
    index: StrictInt
    skill_id: str
    skill_revision: StrictInt | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
SingleAction=Annotated[Union[SetParameter,ReplaceModule,InsertModule,RemoveModule,WrapModule,ReplaceWithSkill,InsertSkill],Field(discriminator='type')]
class Batch(Action):
    type: Literal['batch']='batch'
    actions: list[SingleAction] = Field(min_length=1,max_length=100)
ActionType=Annotated[Union[SetParameter,ReplaceModule,InsertModule,RemoveModule,WrapModule,ReplaceWithSkill,InsertSkill,Batch],Field(discriminator='type')]
parse_action=TypeAdapter(ActionType)

class ActionResult(Record):
    success: bool
    action_id: str
    action_type: str
    revision: int
    committed: bool = False
    changed_nodes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    error_code: str | None = None
    architecture_valid: bool | None = None
    validation: ValidationReport | None = None
    diff: ArchitectureDiff | None = None
