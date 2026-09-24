"""Editable constructor structure, independent of observed ModelIR 0.1."""
from pathlib import Path
from typing import Literal
from pydantic import Field, JsonValue, model_validator
from p2i.ir.model import Record

class TensorContract(Record):
    shape: list[int | str | None] = Field(default_factory=list)
    rank: int | None = Field(default=None, ge=0)
    dtype: str | None = None
    dtype_family: Literal['floating','integer','boolean','complex'] | None = None
    device: str | None = None
    evidence: Literal['observed','declared'] = 'observed'

    @model_validator(mode='after')
    def consistent_rank(self):
        if self.shape:
            if self.rank is not None and self.rank != len(self.shape):raise ValueError('rank must equal shape length')
            self.rank=len(self.shape)
        if any(isinstance(x,int) and x<0 for x in self.shape):raise ValueError('Negative tensor dimension')
        return self

class ConstructorSpec(Record):
    module_type: str
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

class Registration(Record):
    name: str
    node_id: str

class ArchitectureNode(Record):
    id: str
    name: str
    qualified_name: str
    kind: Literal['module','container','custom']
    module_type: str
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    children: list[Registration] = Field(default_factory=list)
    source_observed_module_id: str | None = None
    latest_observed_module_id: str | None = None
    editable: bool
    edit_restrictions: list[str] = Field(default_factory=list)
    managed_by: str | None = None
    input_contracts: list[TensorContract] = Field(default_factory=list)
    output_contracts: list[TensorContract] = Field(default_factory=list)
    contracts_revision: int | None = None
    skill_id: str | None = None
    skill_revision: int | None = None
    training: bool = True

class ArchitectureIR(Record):
    version: Literal['0.1'] = '0.1'
    model_name: str
    root_id: str
    revision: int = 0
    seed: int = 0
    nodes: list[ArchitectureNode]

    @model_validator(mode='after')
    def structure(self):
        nodes={n.id:n for n in self.nodes}
        if len(nodes)!=len(self.nodes): raise ValueError('Duplicate architecture node IDs')
        if self.root_id not in nodes: raise ValueError('Missing root')
        active, visited=set(),set()
        def walk(nid):
            if nid not in nodes: raise ValueError(f'Missing node {nid}')
            if nid in active: raise ValueError(f'Architecture cycle at {nid}')
            if nid in visited:return
            active.add(nid)
            n=nodes[nid]
            names=[r.name for r in n.children]
            if len(names)!=len(set(names)):raise ValueError(f'Duplicate registration names at {nid}')
            if any(not name or '.' in name for name in names):raise ValueError('Invalid registration name')
            if n.managed_by and n.managed_by not in nodes:raise ValueError('Missing managing adapter node')
            for child in n.children:walk(child.node_id)
            active.remove(nid);visited.add(nid)
        walk(self.root_id)
        if visited!=set(nodes):raise ValueError('Unreachable architecture nodes')
        return self

    def save(self,path):
        Path(path).write_text(ArchitectureIR.model_validate(self.model_dump()).model_dump_json(indent=2)+'\n')

    @classmethod
    def load(cls,path):return cls.model_validate_json(Path(path).read_text())

class WeightTransferReport(Record):
    preserved: list[str] = Field(default_factory=list)
    partially_preserved: list[str] = Field(default_factory=list)
    reinitialized: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)

class ValidationIssue(Record):
    severity: Literal['error','warning'] = 'error'
    category: str
    node: str | None = None
    message: str
    suggested_fix: str | None = None
    exception_type: str | None = None
    traceback_summary: str | None = None

class ValidationReport(Record):
    schema_valid: bool = True
    constructor_valid: bool = True
    structural_valid: bool = True
    shape_valid: bool | None = None
    forward_valid: bool | None = None
    backward_valid: bool | None = None
    revision: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)
    @property
    def valid(self):return not any(i.severity=='error' for i in self.issues)

class DiffEntry(Record):
    node_id: str
    path: str
    field: str
    before: JsonValue = None
    after: JsonValue = None

class ArchitectureDiff(Record):
    from_revision: int
    to_revision: int
    changes: list[DiffEntry] = Field(default_factory=list)

def architecture_diff(before,after):
    a,b={n.id:n for n in before.nodes},{n.id:n for n in after.nodes}
    out=ArchitectureDiff(from_revision=before.revision,to_revision=after.revision)
    for nid in sorted(a.keys()|b.keys()):
        if nid not in a or nid not in b:
            n=b.get(nid) or a[nid]
            out.changes.append(DiffEntry(node_id=nid,path=n.qualified_name,field='node',before=a[nid].model_dump(mode='json') if nid in a else None,after=b[nid].model_dump(mode='json') if nid in b else None))
            continue
        for field in ('module_type','parameters','children','skill_id','skill_revision'):
            old,new=a[nid].model_dump(mode='json')[field],b[nid].model_dump(mode='json')[field]
            if field=='parameters':
                for key in sorted(old.keys()|new.keys()):
                    if old.get(key)!=new.get(key):out.changes.append(DiffEntry(node_id=nid,path=b[nid].qualified_name,field=f'parameters.{key}',before=old.get(key),after=new.get(key)))
            elif old!=new:out.changes.append(DiffEntry(node_id=nid,path=b[nid].qualified_name,field=field,before=old,after=new))
    return out
