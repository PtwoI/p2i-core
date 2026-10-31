"""Offline edit -> save -> restore -> train -> save example."""
import argparse

import torch
from torch import nn

import p2i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='model.p2i')
    args = parser.parse_args()
    torch.manual_seed(7)
    inputs = (torch.randn(4, 16),)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    harness = p2i.Harness(model, example_inputs=inputs)
    action = {'type': 'batch', 'actions': [
        {'type': 'set_parameter', 'target': '0', 'parameter': 'out_features', 'value': 24},
        {'type': 'set_parameter', 'target': '2', 'parameter': 'in_features', 'value': 24},
    ]}
    result = harness.apply(action)
    if not result.success:
        raise RuntimeError(result.errors)
    harness.retrace()
    expected = harness.build()(*inputs)
    harness.save(args.output)
    restored = p2i.load_model(args.output)
    torch.testing.assert_close(expected, restored(*inputs), rtol=0, atol=0)
    restored.train()
    optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    optimizer.zero_grad()
    restored(*inputs).square().mean().backward()
    optimizer.step()
    p2i.Harness(restored, example_inputs=inputs).save(args.output)
    print(f'PASS: edited 32 -> 24, restored exact output, trained, saved {args.output}')


if __name__ == '__main__':
    main()
