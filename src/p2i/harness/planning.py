"""Conservative edit suggestions from explicit Sequential registration order."""


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
        family = ('linear' if node.module_type == 'torch.nn.Linear' and parameter == 'out_features'
                  else 'conv1d' if node.module_type == 'torch.nn.Conv1d' and parameter == 'out_channels'
                  else 'conv2d' if node.module_type == 'torch.nn.Conv2d' and parameter == 'out_channels'
                  else None)
        if family and value != node.parameters[parameter]:
            nodes = {n.id: n for n in arch.nodes}
            parents = [p for p in arch.nodes if any(c.node_id == node.id for c in p.children)]
            transparent = {'torch.nn.ReLU', 'torch.nn.GELU', 'torch.nn.SiLU',
                           'torch.nn.Dropout', 'torch.nn.Identity'}
            if family == 'conv2d':
                transparent |= {'torch.nn.MaxPool2d', 'torch.nn.AvgPool2d', 'torch.nn.AdaptiveAvgPool2d'}
            norm = 'torch.nn.BatchNorm1d' if family in ('linear', 'conv1d') else 'torch.nn.BatchNorm2d'
            consumer = {'linear': 'torch.nn.Linear', 'conv1d': 'torch.nn.Conv1d', 'conv2d': 'torch.nn.Conv2d'}[family]
            input_parameter = 'in_features' if family == 'linear' else 'in_channels'
            proven = bool(parents) and all(p.module_type == 'torch.nn.Sequential' and not p.managed_by for p in parents)
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
