"""Opt-in external packages; offline weights, real library forward implementations."""
import os

import pytest
import torch

import p2i
from p2i.checkpoint import checkpoint_bytes


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def external_model(name, device):
    from p2i.verification import make_case
    pytest.importorskip('transformers' if name == 'bert' else 'torchvision')
    return make_case(name, device)


@pytest.mark.parametrize('name', ['resnet18', 'mobilenet_v3_small', 'vit', 'bert'])
def test_external_model_edit_restore_train(name):
    device = os.environ.get('P2I_TEST_DEVICE', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.fail('CUDA verification requested but no CUDA device is available')
    torch.manual_seed(47)
    model, inputs, target = external_model(name, device)
    harness = p2i.Harness(model, example_inputs=inputs)
    observation = harness.retrace()
    assert observation.runtime.calls and observation.operations
    result = harness.apply({'type': 'set_parameter', 'target': target, 'parameter': 'bias', 'value': False})
    assert result.committed, result.errors
    restored = p2i.Harness.load(checkpoint_bytes(harness), model=model, device=device)
    expected, actual = harness.build()(*inputs), restored.build()(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid
    exported = restored.build().train()
    optimizer = torch.optim.SGD(exported.parameters(), lr=0.01)
    weight = exported.get_submodule(target).weight
    before = weight.detach().clone()
    output = exported(*inputs)
    if isinstance(output, tuple):
        output = output[1]
    output.square().mean().backward()
    optimizer.step()
    assert not torch.equal(weight, before)
    assert next(exported.parameters()).device.type == torch.device(device).type


def test_cuda_recurrent_roundtrip():
    device = os.environ.get('P2I_TEST_DEVICE', 'cpu')
    if not device.startswith('cuda'):
        pytest.skip('Set P2I_TEST_DEVICE=cuda to execute CUDA verification')
    if not torch.cuda.is_available():
        pytest.fail('CUDA verification requested but no CUDA device is available')
    for cls in (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU):
        model = cls(4, 8, batch_first=True).to(device).eval()
        inputs = (torch.randn(2, 3, 4, device=device),)
        harness = p2i.Harness(model, example_inputs=inputs)
        restored = p2i.Harness.load(checkpoint_bytes(harness), device=device)
        torch.testing.assert_close(restored.build()(*inputs), model(*inputs))
        assert restored.validate(backward=True).backward_valid
        assert restored.retrace(static=False).operations
    harness = p2i.Harness(torch.nn.Sequential(torch.nn.Linear(4, 4)).to(device),
                          example_inputs=(torch.randn(2, 4, device=device),))
    result = harness.apply({'type': 'insert_module', 'parent': '', 'index': 1,
                            'module': {'module_type': 'torch.nn.Linear', 'parameters': {'in_features': 4, 'out_features': 2}}})
    assert result.committed, result.errors
    assert all(parameter.is_cuda for parameter in harness.build().parameters())
