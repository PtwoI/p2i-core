"""Versioned metadata-only IR. Hierarchy edges are not computation edges."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

class CodeOrigin(Record):
    kind: Literal["user", "framework", "dependency", "unknown"] = "unknown"
    package: str | None = None
    package_version: str | None = None
    file: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    module_name: str | None = None
    class_name: str | None = None

class SourceLocation(Record):
    file: str
    line: int | None = None
    function: str | None = None

class ModuleNode(Record):
    id: str
    name: str
    qualified_name: str
    aliases: list[str] = Field(default_factory=list)
    class_name: str
    module_path: str
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    origin: CodeOrigin
    parameter_count: int | None = None
    trainable_parameter_count: int | None = None
    input_tensor_ids: list[str] = Field(default_factory=list)
    output_tensor_ids: list[str] = Field(default_factory=list)
    executed: bool = False
    call_ids: list[str] = Field(default_factory=list)

class ModuleEdge(Record):
    parent_id: str
    child_id: str
    name: str
    shared: bool = False

class TensorInfo(Record):
    id: str
    graph: Literal["runtime", "export", "fx"] = "runtime"
    shape: list[int | str | None]
    dtype: str | None = None
    device: str | None = None
    requires_grad: bool | None = None
    producer_id: str | None = None
    consumer_ids: list[str] = Field(default_factory=list)
    alias_of: str | None = None

class OperationNode(Record):
    id: str
    graph: Literal["runtime", "export", "fx"]
    op_type: str
    display_name: str
    parent_module_id: str | None = None
    input_tensor_ids: list[str] = Field(default_factory=list)
    output_tensor_ids: list[str] = Field(default_factory=list)
    source: SourceLocation | None = None
    executed: bool = False
    call_id: str | None = None
    sequence: int

class DataEdge(Record):
    tensor_id: str
    source_id: str
    target_id: str

class RuntimeCall(Record):
    id: str
    module_id: str
    parent_call_id: str | None = None
    sequence: int
    input_tensor_ids: list[str] = Field(default_factory=list)
    output_tensor_ids: list[str] = Field(default_factory=list)
    completed: bool = False

class AnalysisStatus(Record):
    technique: str
    status: Literal["success", "partial", "failed", "skipped"]
    message: str | None = None

class RuntimeMetadata(Record):
    calls: list[RuntimeCall] = Field(default_factory=list)
    input_tensor_ids: list[str] = Field(default_factory=list)
    output_tensor_ids: list[str] = Field(default_factory=list)
    grad_enabled: bool = False

class ModelMetadata(Record):
    name: str
    torch_version: str
    training: bool
    analysis: list[AnalysisStatus] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class ModelIR(Record):
    version: Literal["0.1"] = "0.1"
    metadata: ModelMetadata
    modules: list[ModuleNode] = Field(default_factory=list)
    operations: list[OperationNode] = Field(default_factory=list)
    tensors: list[TensorInfo] = Field(default_factory=list)
    module_edges: list[ModuleEdge] = Field(default_factory=list)
    data_edges: list[DataEdge] = Field(default_factory=list)
    runtime: RuntimeMetadata | None = None

    @model_validator(mode="after")
    def references(self):
        def unique(items):
            ids = [x.id for x in items]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate IDs")
            return set(ids)
        ms, ops, ts = unique(self.modules), unique(self.operations), unique(self.tensors)
        calls = unique(self.runtime.calls) if self.runtime else set()
        def require(ok, message):
            if not ok:
                raise ValueError(message)
        module_map = {m.id: m for m in self.modules}
        hierarchy_pairs = {(e.parent_id, e.child_id) for e in self.module_edges}
        for m in self.modules:
            require(m.parent_id is None or (m.parent_id, m.id) in hierarchy_pairs, "Missing canonical parent edge")
            require(all((m.id, c) in hierarchy_pairs for c in m.children), "Missing child edge")
            require(m.parent_id is None or m.parent_id in ms, "Invalid module parent")
            require(set(m.children) <= ms, "Invalid module children")
            require(set(m.call_ids) <= calls, "Invalid module call")
        for e in self.module_edges:
            require(e.parent_id in ms and e.child_id in ms, "Invalid hierarchy edge")
            parent = module_map[e.parent_id]
            require(e.child_id in parent.children, "Hierarchy edge missing from children")
        for o in self.operations:
            require(o.parent_module_id is None or o.parent_module_id in ms, "Invalid operation module")
            require(o.call_id is None or o.call_id in calls, "Invalid operation call")
        for obj in [*self.modules, *self.operations, *(self.runtime.calls if self.runtime else [])]:
            require(set(obj.input_tensor_ids + obj.output_tensor_ids) <= ts, "Invalid tensor reference")
        if self.runtime:
            require(set(self.runtime.input_tensor_ids + self.runtime.output_tensor_ids) <= ts, "Invalid model IO")
            for c in self.runtime.calls:
                require(c.module_id in ms and (c.parent_call_id is None or c.parent_call_id in calls), "Invalid runtime call")
        opmap = {o.id: o for o in self.operations}
        for t in self.tensors:
            require(t.producer_id is None or t.producer_id in ops, "Invalid producer")
            require(set(t.consumer_ids) <= ops, "Invalid consumers")
            require(t.alias_of is None or t.alias_of in ts, "Invalid alias")
            if t.producer_id:
                require(t.id in opmap[t.producer_id].output_tensor_ids, "Producer mismatch")
                require(t.graph == opmap[t.producer_id].graph, "Cross-graph producer")
            for c in t.consumer_ids:
                require(t.id in opmap[c].input_tensor_ids, "Consumer mismatch")
                require(t.graph == opmap[c].graph, "Cross-graph consumer")
        expected = {(t.id, t.producer_id, c) for t in self.tensors if t.producer_id for c in t.consumer_ids}
        require({(e.tensor_id, e.source_id, e.target_id) for e in self.data_edges} == expected, "Invalid data edges")
        return self

    def save(self, path: str | Path) -> None:
        # Revalidate mutable objects before writing.
        type(self).model_validate(self.model_dump())
        Path(path).write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
