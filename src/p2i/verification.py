"""Offline, reproducible device verification for installed P2I packages."""
import argparse
import json
from pathlib import Path
import platform
import torch
from torch import nn

DEMOS = ('mlp', 'cnn', 'transformer', 'rnn', 'lstm', 'gru')
EXTERNAL = ('resnet18', 'mobilenet_v3_small', 'vit', 'bert')


def make_case(name, device='cpu'):
    from p2i.examples import make_demo
    if name in DEMOS:
        model, inputs = make_demo(name)
        target = {'mlp': 'layers.0', 'cnn': 'features.0'}.get(name, 'head')
    elif name == 'bert':
        from transformers import BertConfig, BertModel
        model = BertModel(BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, return_dict=False, attn_implementation='eager'))
        inputs, target = (torch.randint(0, 64, (2, 6)),), 'pooler.dense'
    elif name == 'vit':
        from torchvision.models.vision_transformer import VisionTransformer
        model = VisionTransformer(image_size=32, patch_size=8, num_layers=1, num_heads=4,
                                  hidden_dim=32, mlp_dim=64, num_classes=4)
        nn.init.normal_(model.heads.head.weight, std=0.02)
        inputs, target = (torch.randn(2, 3, 32, 32),), 'heads.head'
    elif name in ('resnet18', 'mobilenet_v3_small'):
        from torchvision import models
        model = getattr(models, name)(weights=None, num_classes=4)
        inputs, target = (torch.randn(2, 3, 32, 32),), 'fc' if name == 'resnet18' else 'classifier.3'
    else:
        raise ValueError(f'Unknown verification model: {name}')
    return model.to(device).eval(), tuple(x.to(device) for x in inputs), target


def verify(device='cpu', models=DEMOS):
    import p2i
    from p2i.checkpoint import checkpoint_bytes
    report = {'format': 'p2i.verification.v1', 'p2i': p2i.__version__, 'torch': str(torch.__version__),
              'python': platform.python_version(), 'device': device, 'cuda_available': torch.cuda.is_available(),
              'gpu': None, 'success': False, 'cases': []}
    if device == 'cuda':
        if not torch.cuda.is_available():
            report['error'] = 'CUDA was requested but no CUDA device is available. No GPU tests ran.'
            return report
        report['gpu'] = torch.cuda.get_device_name(0)
    elif device != 'cpu':
        report['error'] = 'Device must be cpu or cuda'
        return report
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device == 'cuda' else []):
        for name in models:
            result = {'model': name, 'success': False}
            try:
                torch.manual_seed(47)
                model, inputs, target = make_case(name, device)
                harness = p2i.Harness(model, example_inputs=inputs)
                edited = harness.apply({'type': 'set_parameter', 'target': target, 'parameter': 'bias', 'value': False})
                if not edited.committed:
                    raise ValueError('; '.join(edited.errors))
                observed = harness.retrace(static=False)
                restored = p2i.Harness.load(checkpoint_bytes(harness), model=model, device=device)
                torch.testing.assert_close(harness.build()(*inputs), restored.build()(*inputs), rtol=0, atol=0)
                validation = restored.validate(backward=True)
                if not validation.backward_valid:
                    raise ValueError('; '.join(issue.message for issue in validation.issues))
                exported = restored.build().train()
                parameter = exported.get_submodule(target).weight
                before = parameter.detach().clone()
                optimizer = torch.optim.SGD(exported.parameters(), lr=0.01)
                output = exported(*inputs)
                if isinstance(output, tuple):
                    output = output[1]
                output.square().mean().backward()
                optimizer.step()
                if torch.equal(before, parameter):
                    raise ValueError('The optimizer did not update the selected weight')
                result.update(success=True, operations=len(observed.operations),
                              backward=True, checkpoint=True, optimizer=True)
            except Exception as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
            report['cases'].append(result)
    report['success'] = bool(report['cases']) and all(case['success'] for case in report['cases'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--external', action='store_true', help='Include offline TorchVision and BERT configurations')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    report = verify(args.device, DEMOS + EXTERNAL if args.external else DEMOS)
    data = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(data + '\n')
    print(data)
    return 0 if report['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
