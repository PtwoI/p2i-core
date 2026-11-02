"""Revision-bound suggestions from Sequential order and explicit architecture rules."""
from torch import nn


def sequential_parent(harness, parent):
    original = harness._originals.get(parent.id)
    return parent.module_type == 'torch.nn.Sequential' or (
        isinstance(original, nn.Sequential) and type(original).forward is nn.Sequential.forward)


def architecture_followers(harness, arch, node, parameter, value):
    """Known forward implementations only; no inference from naming alone."""
    nodes = {item.id: item for item in arch.nodes}
    parents = [p for p in arch.nodes if any(c.node_id == node.id for c in p.children)]
    if len(parents) != 1:
        return None
    parent = parents[0]
    children = {c.name: nodes[c.node_id] for c in parent.children}
    if parent.module_type in ('torchvision.models.resnet.BasicBlock', 'torchvision.models.resnet.Bottleneck'):
        mapping = {'conv1': ('bn1', 'conv2'), 'conv2': ('bn2', 'conv3')}
        name = next((key for key, child in children.items() if child.id == node.id), '')
        if parameter == 'out_channels' and name in mapping:
            norm, consumer = mapping[name]
            if norm in children and consumer in children:
                return [(children[norm], 'num_features', value), (children[consumer], 'in_channels', value)], 'Residual-block hidden width; the residual output width is preserved.'
    if parent.module_type == 'transformers.models.bert.modeling_bert.BertIntermediate' and parameter == 'out_features' and children.get('dense') is node:
        owners = [p for p in arch.nodes if any(c.node_id == parent.id for c in p.children)]
        if len(owners) == 1 and owners[0].module_type == 'transformers.models.bert.modeling_bert.BertLayer':
            output = next((nodes[c.node_id] for c in owners[0].children if c.name == 'output'), None)
            if output and output.module_type == 'transformers.models.bert.modeling_bert.BertOutput':
                dense = next((nodes[c.node_id] for c in output.children if c.name == 'dense'), None)
                if dense:
                    return [(dense, 'in_features', value)], 'BERT feed-forward hidden width; the residual output width is preserved.'
    if parent.module_type == 'p2i.examples.recurrent.RecurrentClassifier' and children.get('recurrent') is node and parameter in ('hidden_size', 'bidirectional', 'proj_size'):
        head = children.get('head')
        config = {**node.parameters, parameter: value}
        if head and head.module_type == 'torch.nn.Linear':
            width = (config.get('proj_size') or config['hidden_size']) * (2 if config['bidirectional'] else 1)
            return [(head, 'in_features', width)], 'Recurrent output width includes projection and direction count.'
    return None


def plan_parameter_edit(harness, target, parameter, value, expected_revision=None):
    with harness._lock:
        if type(target) is not str or type(parameter) is not str:
            raise ValueError('Target and parameter must be strings')
        if expected_revision is not None and type(expected_revision) is not int:
            raise ValueError('Expected revision must be an integer')
        arch = harness.architecture()
        if expected_revision is not None and expected_revision != arch.revision:
            raise ValueError('Stale revision')
        node = harness._resolve(arch, target)
        if not node.editable or node.managed_by or parameter not in node.parameters:
            raise ValueError('Choose an editable constructor parameter')
        parameters = {**node.parameters, parameter: value}
        harness.registry.get(node.module_type).validate(parameters)
        changes = {(node.id, parameter): value}
        notes, unresolved = [], []
        architecture_rule = architecture_followers(harness, arch, node, parameter, value)
        if architecture_rule and value != node.parameters[parameter]:
            followers, reason = architecture_rule
            for other, key, proposed in followers:
                registrations = sum(c.node_id == other.id for p in arch.nodes for c in p.children)
                if registrations != 1 or not other.editable or other.managed_by:
                    raise ValueError('Shared or managed architecture followers require an explicit batch')
                harness.registry.get(other.module_type).validate({**other.parameters, key: proposed})
                changes[(other.id, key)] = proposed
                notes.append(f'Update {other.qualified_name}.{key} to {proposed}. {reason}')
        family = ('linear' if node.module_type == 'torch.nn.Linear' and parameter == 'out_features'
                  else 'conv1d' if node.module_type == 'torch.nn.Conv1d' and parameter == 'out_channels'
                  else 'conv2d' if node.module_type == 'torch.nn.Conv2d' and parameter == 'out_channels'
                  else None)
        if family and not architecture_rule and value != node.parameters[parameter]:
            nodes = {n.id: n for n in arch.nodes}
            parents = [p for p in arch.nodes if any(c.node_id == node.id for c in p.children)]
            transparent = {'torch.nn.ReLU', 'torch.nn.GELU', 'torch.nn.SiLU',
                           'torch.nn.Dropout', 'torch.nn.Identity'}
            if family == 'conv2d':
                transparent |= {'torch.nn.MaxPool2d', 'torch.nn.AvgPool2d', 'torch.nn.AdaptiveAvgPool2d'}
            norm = 'torch.nn.BatchNorm1d' if family in ('linear', 'conv1d') else 'torch.nn.BatchNorm2d'
            consumer = {'linear': 'torch.nn.Linear', 'conv1d': 'torch.nn.Conv1d', 'conv2d': 'torch.nn.Conv2d'}[family]
            input_parameter = 'in_features' if family == 'linear' else 'in_channels'
            proven = bool(parents) and all(sequential_parent(harness, p) and not p.managed_by for p in parents)
            if not proven:
                unresolved.append('Connections outside Sequential require explicit batch edits; no dependency is inferred from names or trace order.')
            else:
                for parent in parents:
                    for i, registration in enumerate(parent.children):
                        if registration.node_id != node.id:
                            continue
                        for following in parent.children[i + 1:]:
                            other = nodes[following.node_id]
                            key = ('num_features' if other.module_type == norm
                                   else input_parameter if other.module_type == consumer else None)
                            if key:
                                if other.parameters[key] != node.parameters[parameter]:
                                    unresolved.append(f'{other.qualified_name}: existing dimensions do not match; edit explicitly.')
                                    break
                                # A shared consumer may have other, incompatible callers.
                                registrations = sum(c.node_id == other.id for p in arch.nodes for c in p.children)
                                if other.id == node.id or registrations != 1 or other.managed_by:
                                    unresolved.append(f'{other.qualified_name}: shared modules require explicit batch edits.')
                                    break
                                changes[(other.id, key)] = value
                                notes.append(f'Update {other.qualified_name}.{key} to {value} along Sequential order.')
                                if other.module_type == consumer:
                                    break
                            elif other.module_type not in transparent:
                                unresolved.append(f'Stopped at {other.qualified_name}; its dimension mapping requires an explicit edit.')
                                break
        actions = [{'type': 'set_parameter', 'target': nid, 'parameter': key, 'value': v}
                   for (nid, key), v in changes.items()]
        return {'success': True, 'action': {'type': 'batch', 'expected_revision': arch.revision, 'actions': actions},
                'notes': notes, 'unresolved': unresolved}
