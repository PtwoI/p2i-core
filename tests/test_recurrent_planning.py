from io import BytesIO

import pytest
import torch
from torch import nn

import p2i
from p2i.checkpoint import checkpoint_bytes
from p2i.examples import make_demo


@pytest.mark.parametrize('cls', [nn.RNN, nn.LSTM, nn.GRU])
@pytest.mark.parametrize('bidirectional', [False, True])
def test_recurrent_roundtrip_and_edit(cls, bidirectional):
    kwargs = dict(input_size=4, hidden_size=6, num_layers=2, batch_first=True,
                  bidirectional=bidirectional, dropout=0.1)
    if cls is nn.LSTM:
        kwargs['proj_size'] = 3
    if cls is nn.RNN:
        kwargs['nonlinearity'] = 'relu'
    model = cls(**kwargs).double().eval()
    x = torch.randn(2, 3, 4, dtype=torch.double)
    harness = p2i.Harness(model, example_inputs=(x,))
    restored = p2i.Harness.load(checkpoint_bytes(harness))
    before, after = model(x), restored.build()(x)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid
    assert restored.apply({'type': 'set_parameter', 'target': '', 'parameter': 'hidden_size', 'value': 8}).committed
    assert restored.build().hidden_size == 8
    assert restored.retrace(static=False).runtime.calls


@pytest.mark.parametrize('name', ['rnn', 'lstm', 'gru'])
def test_recurrent_demo_train_checkpoint(name):
    model, inputs = make_demo(name)
    harness = p2i.Harness(model, example_inputs=inputs)
    result = harness.apply({'type': 'batch', 'actions': [
        {'type': 'set_parameter', 'target': 'recurrent', 'parameter': 'hidden_size', 'value': 12},
        {'type': 'set_parameter', 'target': 'head', 'parameter': 'in_features', 'value': 12},
    ]})
    assert result.committed
    exported = p2i.load_model(checkpoint_bytes(harness))
    assert exported.recurrent.hidden_size == exported.head.in_features == 12
    optimizer = torch.optim.SGD(exported.parameters(), lr=0.1)
    before = exported.head.weight.detach().clone()
    exported(*inputs).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, exported.head.weight)
    assert harness.validate(backward=True).backward_valid


@pytest.mark.parametrize('parameter,value', [('hidden_size', 0), ('num_layers', True),
    ('bidirectional', 1), ('proj_size', -1), ('proj_size', 8), ('proj_size', 1.5)])
def test_recurrent_invalid_edit_preserves_state(parameter, value):
    harness = p2i.Harness(nn.LSTM(4, 8), example_inputs=(torch.randn(2, 1, 4),))
    assert not harness.apply({'type': 'set_parameter', 'target': '', 'parameter': parameter, 'value': value}).success
    assert harness.architecture().revision == 0


def test_sequential_dimension_plan_is_atomic_and_revision_bound():
    harness = p2i.Harness(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(), nn.Linear(8, 2)).eval(),
                          example_inputs=(torch.randn(2, 4),))
    plan = harness.plan_edit('0', 'out_features', 6, expected_revision=0)
    assert len(plan['action']['actions']) == 2
    assert harness.architecture().revision == 0
    assert harness.preview(plan['action']).success
    assert harness.apply(plan['action']).committed
    assert harness.build()[3].in_features == 6
    assert not harness.apply(plan['action']).success
    with pytest.raises(ValueError, match='Stale'):
        harness.plan_edit('0', 'out_features', 5, expected_revision=0)


def test_conv_plan_includes_norm_and_downstream_consumer():
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(),
                          nn.MaxPool2d(2), nn.Conv2d(8, 4, 1)).eval()
    harness = p2i.Harness(model, example_inputs=(torch.randn(2, 3, 12, 12),))
    plan = harness.plan_edit('0', 'out_channels', 6)
    assert len(plan['action']['actions']) == 3
    assert harness.apply(plan['action']).committed
    assert harness.build()[1].num_features == harness.build()[4].in_channels == 6
    assert harness.validate(backward=True).backward_valid


def test_plan_stops_at_unknown_mapping_and_shared_consumers():
    for model in [nn.Sequential(nn.Linear(4, 8), nn.Flatten(), nn.Linear(8, 2)),
                  nn.Sequential(nn.Linear(4, 8), *([nn.Linear(8, 8)] * 2))]:
        harness = p2i.Harness(model)
        plan = harness.plan_edit('0', 'out_features', 6)
        assert len(plan['action']['actions']) == 1
        assert plan['unresolved']
    model, inputs = make_demo('lstm')
    harness = p2i.Harness(model, example_inputs=inputs)
    assert len(harness.plan_edit('head', 'out_features', 5)['action']['actions']) == 1


def test_inserted_module_inherits_model_dtype():
    harness = p2i.Harness(nn.Sequential(nn.Linear(4, 4)).double(),
                          example_inputs=(torch.randn(2, 4, dtype=torch.double),))
    result = harness.apply({'type': 'insert_module', 'parent': '', 'index': 1,
                            'module': {'module_type': 'torch.nn.Linear', 'parameters': {'in_features': 4, 'out_features': 2}}})
    assert result.committed, result.errors
    assert all(parameter.dtype == torch.double for parameter in harness.build().parameters())
