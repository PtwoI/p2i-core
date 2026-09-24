"""Opt-in, local study records. No telemetry, automatic grading, or participant identity."""
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class TaskRecord(BaseModel):
    version: Literal['0.1']='0.1'
    task_id: str
    task_kind: Literal['comprehension','navigation','fault_localization','comparison','debugging','edit_consequence']
    observed_revision: int | None=None
    elapsed_seconds: float=Field(ge=0)
    navigation_actions: int=Field(ge=0)
    correct: bool | None=None
    confidence: int | None=Field(default=None,ge=1,le=5)
    timestamp: str
    notes: str=''


class TaskRecorder:
    """A study controller explicitly starts, counts and finishes each task.

    Correctness is supplied by the study rubric, never inferred from speed.
    Only task-level metadata is stored; no tensor payloads or user identifiers.
    """
    def __init__(self):
        self.records=[];self._active=None
    def start(self,task_id,task_kind,*,observed_revision=None):
        if self._active is not None:raise ValueError('Finish the active task first')
        # Validate identifiers and category before starting the clock.
        TaskRecord(task_id=task_id,task_kind=task_kind,observed_revision=observed_revision,elapsed_seconds=0,navigation_actions=0,timestamp='')
        self._active=dict(task_id=task_id,task_kind=task_kind,observed_revision=observed_revision);self._start=time.monotonic();self._actions=0
    def navigation(self):
        if self._active is None:raise ValueError('No active task')
        self._actions+=1
    def finish(self,*,correct=None,confidence=None,notes=''):
        if self._active is None:raise ValueError('No active task')
        record=TaskRecord(**self._active,elapsed_seconds=time.monotonic()-self._start,navigation_actions=self._actions,correct=correct,confidence=confidence,timestamp=datetime.now(timezone.utc).isoformat(),notes=notes)
        self.records.append(record);self._active=None;return record.model_copy(deep=True)
    def save(self,path):
        Path(path).write_text('\n'.join(r.model_dump_json() for r in self.records)+'\n',encoding='utf-8')
