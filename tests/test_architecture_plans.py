import pytest
import torch
import p2i
from p2i.checkpoint import checkpoint_bytes
from p2i.verification import make_case, verify


@pytest.mark.parametrize('name,target,parameter,value,consumer,consumer_parameter', [
    ('resnet18', 'layer1.0.conv1', 'out_channels', 48, 'layer1.0.conv2', 'in_channels'),
    ('bert', 'encoder.layer.0.intermediate.dense', 'out_features', 24, 'encoder.layer.0.output.dense', 'in_features'),
    ('vit', 'encoder.layers.encoder_layer_0.mlp.0', 'out_features', 48, 'encoder.layers.encoder_layer_0.mlp.3', 'in_features'),
    ('vit', 'encoder.layers.encoder_layer_0.self_attention', 'num_heads', 2, 'encoder.layers.encoder_layer_0.self_attention', 'num_heads'),
])
def test_known_architecture_plan_roundtrip(name, target, parameter, value, consumer, consumer_parameter):
    pytest.importorskip('transformers' if name == 'bert' else 'torchvision')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, inputs, _ = make_case(name)
        harness = p2i.Harness(model, example_inputs=inputs)
        plan = harness.plan_edit(target, parameter, value)
        assert harness.preview(plan['action']).success
        result = harness.apply(plan['action'])
        assert result.committed, result.errors
        assert getattr(harness.build().get_submodule(consumer), consumer_parameter) == value
        assert harness.validate(backward=True).backward_valid
        restored = p2i.Harness.load(checkpoint_bytes(harness), model=model)
        torch.testing.assert_close(harness.build()(*inputs), restored.build()(*inputs), rtol=0, atol=0)
        assert restored.retrace(static=False).operations
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('name,parameter,value,expected', [('rnn', 'hidden_size', 12, 12),
    ('lstm', 'proj_size', 8, 8), ('gru', 'bidirectional', True, 32)])
def test_recurrent_output_plan(name, parameter, value, expected):
    model, inputs, _ = make_case(name)
    harness = p2i.Harness(model, example_inputs=inputs)
    plan = harness.plan_edit('recurrent', parameter, value)
    assert len(plan['action']['actions']) == 2
    assert harness.apply(plan['action']).committed
    assert harness.build().head.in_features == expected
    assert harness.validate(backward=True).backward_valid


def test_verification_reports_actual_execution_and_missing_cuda():
    report = verify('cpu', models=('mlp', 'lstm'))
    assert report['success'] and len(report['cases']) == 2
    assert all(case['optimizer'] and case['checkpoint'] for case in report['cases'])
    if not torch.cuda.is_available():
        unavailable = verify('cuda')
        assert not unavailable['success'] and unavailable['cases'] == []
        assert 'No GPU tests ran' in unavailable['error']
