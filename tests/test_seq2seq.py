import pytest
import torch
import p2i
from p2i.examples import make_demo
from p2i.checkpoint import checkpoint_bytes


def test_encoder_decoder_trace_edit_and_independent_restore():
    torch.set_num_threads(1)
    model, inputs = make_demo('seq2seq')
    harness = p2i.Harness(model, example_inputs=inputs)
    observed = harness.retrace(static=False)
    assert observed.operations
    nodes = {n.qualified_name: n for n in harness.architecture().nodes}
    assert nodes['decoder.0'].editable and nodes['encoder.0'].editable
    assert nodes['decoder.0.multihead_attn'].managed_by == nodes['decoder.0'].id
    assert nodes['decoder.0'].parameters['nhead'] == 4
    assert harness.apply(dict(type='set_parameter', target='decoder.0', parameter='dim_feedforward', value=48)).committed
    assert harness.build().decoder[0].linear1.out_features == 48
    assert next(n for n in harness.architecture().nodes if n.qualified_name == 'decoder.0.linear1').parameters['out_features'] == 48
    restored = p2i.Harness.load(checkpoint_bytes(harness))
    torch.testing.assert_close(harness.build()(*inputs), restored.build()(*inputs), rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid
    assert restored.retrace(static=False).operations
    rejected = restored.apply(dict(type='set_parameter', target='decoder.0', parameter='nhead', value=3))
    assert not rejected.committed


def test_decoder_is_causal_and_encoder_memory_is_used():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model, (source, target) = make_demo('seq2seq')
    expected = model(source, target)
    changed = target.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 32
    torch.testing.assert_close(expected[:, :-1], model(source, changed)[:, :-1])
    assert not torch.allclose(expected, model((source + 1) % 32, target))
    permuted = target.flip(1)
    assert not torch.allclose(expected.flip(1), model(source, permuted))


def test_transformer_constructor_defaults_are_json_portable():
    from p2i.skills.registry import builtin_specs
    from p2i.architecture.registry import registry
    specs = list(builtin_specs())
    for spec in specs:
        if spec.name == 'TransformerDecoderLayer':
            assert next(p.default for p in spec.hyperparameters if p.name == 'activation') == 'relu'
    module = registry.get('torch.nn.TransformerEncoderLayer').build({'d_model':16,'nhead':2})
    assert module.linear1.out_features == 2048
