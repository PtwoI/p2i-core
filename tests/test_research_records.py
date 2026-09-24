import json
import pytest
from p2i.research import TaskRecorder


def test_opt_in_task_records_do_not_invent_correctness(tmp_path):
    recorder=TaskRecorder()
    with pytest.raises(ValueError):recorder.navigation()
    recorder.start('find-producer','navigation',observed_revision=2)
    recorder.navigation();recorder.navigation()
    with pytest.raises(ValueError):recorder.start('other','debugging')
    result=recorder.finish(confidence=3)
    assert result.correct is None and result.navigation_actions==2
    recorder.save(tmp_path/'study.jsonl')
    assert json.loads((tmp_path/'study.jsonl').read_text())['observed_revision']==2


def test_invalid_task_metadata_does_not_consume_active_task():
    recorder=TaskRecorder();recorder.start('t','comparison')
    with pytest.raises(ValueError):recorder.finish(confidence=6)
    assert recorder.finish(correct=False).correct is False
