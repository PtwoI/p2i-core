from io import BytesIO

import pytest
import torch
from torch import nn

import p2i
from p2i.checkpoint import checkpoint_bytes
from p2i.examples import make_demo


@pytest.mark.parametrize('name', ['mlp', 'cnn', 'transformer'])
def test_edit_save_reload_retrace_and_train(name, tmp_path):
    torch.manual_seed(9)
    model, inputs = make_demo(name)
    harness = p2i.Harness(model, example_inputs=inputs)
    target = {'mlp': 'layers.0', 'cnn': 'features.0', 'transformer': 'head'}[name]
    assert harness.apply({'type': 'set_parameter', 'target': target,
                          'parameter': 'bias', 'value': False}).committed
    before = harness.build()
    expected = before(*inputs)
    path = harness.save(tmp_path / f'{name}.p2i')
    restored = p2i.Harness.load(path)
    assert restored.observe()['revision'] == 1
    assert restored.observe()['baseline_revision'] == 1
    assert restored.observe()['observed_revision'] is None
    assert restored.history() == []
    assert restored.validate(backward=True).backward_valid
    assert restored.retrace(static=False).modules
    actual = restored.build()
    torch.testing.assert_close(actual(*inputs), expected, rtol=0, atol=0)
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in before.state_dict().items())
    assert restored.apply({'type': 'set_parameter', 'target': target,
                           'parameter': 'bias', 'value': True}).committed
    assert restored.undo()['success']
    assert restored.build().get_submodule(target).bias is None
    # The exported object is a normal module, independent of the editing session.
    exported = p2i.load_model(path)
    optimizer = torch.optim.SGD(exported.parameters(), lr=0.01)
    old = exported.get_submodule(target).weight.detach().clone()
    exported.train()
    optimizer.zero_grad()
    exported(*inputs).square().mean().backward()
    optimizer.step()
    assert not torch.equal(old, exported.get_submodule(target).weight)
    # Re-enter P2I with the trained module; new learned weights survive too.
    p2i.Harness(exported, example_inputs=inputs).save(path)
    torch.testing.assert_close(p2i.load_model(path).get_submodule(target).weight,
                               exported.get_submodule(target).weight, rtol=0, atol=0)


class UserModel(nn.Module):
    def __init__(self, scale=3):
        super().__init__()
        self.scale = scale
        self.layers = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        self.register_buffer('offset', torch.tensor(0.25), persistent=False)

    def forward(self, x, *, shift=0):
        return self.layers(x) * self.scale + self.offset + shift


def test_user_model_template_and_kwargs(tmp_path):
    model = UserModel()
    harness = p2i.Harness(model, example_inputs=(torch.randn(2, 4),), example_kwargs={'shift': 0.5})
    assert harness.apply({'type': 'batch', 'actions': [
        {'type': 'set_parameter', 'target': 'layers.0', 'parameter': 'out_features', 'value': 6},
        {'type': 'set_parameter', 'target': 'layers.2', 'parameter': 'in_features', 'value': 6},
    ]}).committed
    checkpoint = harness.save(tmp_path / 'user.p2i')
    with pytest.raises(ValueError, match='requires model='):
        p2i.Harness.load(checkpoint)
    template = UserModel()
    template.offset.fill_(100)  # Saved non-persistent buffers also win over templates.
    restored = p2i.Harness.load(checkpoint, model=template)
    assert restored.validate(backward=True).backward_valid
    x = torch.randn(2, 4)
    torch.testing.assert_close(restored.build()(x, shift=0.5), harness.build()(x, shift=0.5), rtol=0, atol=0)
    assert template.layers[0].out_features == 8  # Loading never mutates the template.
    with pytest.raises(ValueError, match='Template type mismatch'):
        p2i.Harness.load(checkpoint, model=nn.Linear(4, 2))


def test_modes_dtypes_shared_modules_and_tied_weights(tmp_path):
    shared = nn.Linear(4, 4).double()
    other = nn.Linear(4, 4).double()
    other.weight = shared.weight
    model = nn.Sequential(shared, shared, nn.Dropout(0.2), other).eval()
    model[2].train()  # Per-node modes are independent.
    shared.weight.requires_grad_(False)
    harness = p2i.Harness(model, example_inputs=(torch.ones(2, 4, dtype=torch.float64),))
    path = harness.save(tmp_path / 'tied.p2i')
    restored = p2i.load_model(path)
    assert restored[0] is restored[1]
    assert restored[0].weight is restored[3].weight
    assert restored[0].weight.dtype == torch.float64
    assert not restored[0].weight.requires_grad
    assert restored.training is False and restored[2].training is True


def test_shared_batchnorm_buffers_remain_shared(tmp_path):
    left, right = nn.BatchNorm1d(4), nn.BatchNorm1d(4)
    right.running_mean = left.running_mean
    model = nn.Sequential(left, right).eval()
    harness = p2i.Harness(model, example_inputs=(torch.randn(2, 4),))
    path = harness.save(tmp_path / 'buffers.p2i')
    restored = p2i.load_model(path)
    assert restored[0].running_mean is restored[1].running_mean


def test_inputs_opt_out_and_rng_isolation(tmp_path):
    harness = p2i.Harness(nn.Linear(4, 2), example_inputs=(torch.randn(1, 4),))
    rng = torch.get_rng_state().clone()
    path = harness.save(tmp_path / 'private.p2i', include_inputs=False)
    loaded = p2i.Harness.load(path)
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match='example_inputs'):
        loaded.retrace()
    assert loaded.build()(torch.ones(1, 4)).shape == (1, 2)


def test_invalid_format_and_no_module_pickle(tmp_path):
    with pytest.raises(ValueError, match='weights-only'):
        p2i.Harness.load(b'not a checkpoint')
    stream = BytesIO()
    torch.save({'format': 'other', 'version': 1}, stream)
    with pytest.raises(ValueError, match='Not a supported'):
        p2i.Harness.load(stream.getvalue())
    stream = BytesIO()
    torch.save(nn.Linear(2, 2), stream)
    with pytest.raises(ValueError, match='weights-only'):
        p2i.Harness.load(stream.getvalue())
    harness = p2i.Harness(nn.Linear(2, 2), example_inputs=(torch.ones(1, 2),))
    raw = checkpoint_bytes(harness)
    data = torch.load(BytesIO(raw), weights_only=True)
    assert data['format'] == 'p2i.checkpoint'
    assert data['example_inputs'][0].device.type == 'cpu'
    path = tmp_path / 'atomic.p2i'
    path.write_bytes(raw)
    harness._args = (object(),)
    with pytest.raises(ValueError, match='Unsupported checkpoint value'):
        harness.save(path)
    assert path.read_bytes() == raw
