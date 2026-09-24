from functools import lru_cache
from p2i.ir import ModuleNode, ModuleEdge
from .source import origin

def extract(model):
    nodes, edges, identities, paths = [], [], {}, {}
    cached_origin = lru_cache(None)(origin)
    def visit(module, path, parent=None, edge_name=""):
        shared = id(module) in identities
        if shared:
            mid = identities[id(module)]
            node = next(n for n in nodes if n.id == mid)
            if path not in node.aliases:
                node.aliases.append(path)
        else:
            mid = f"m{len(nodes)}"
            identities[id(module)] = mid
            try:
                params = list(module.parameters())
                total = sum(p.numel() for p in params)
                trainable = sum(p.numel() for p in params if p.requires_grad)
            except ValueError:  # Lazy / uninitialized parameters
                total = trainable = None
            node = ModuleNode(id=mid, name=path.rsplit('.', 1)[-1] or type(module).__name__,
                              qualified_name=path, class_name=type(module).__qualname__,
                              module_path=type(module).__module__, parent_id=parent,
                              origin=cached_origin(type(module)), parameter_count=total,
                              trainable_parameter_count=trainable)
            nodes.append(node)
        paths[path] = mid
        if parent:
            edges.append(ModuleEdge(parent_id=parent, child_id=mid, name=edge_name, shared=shared))
            p = next(n for n in nodes if n.id == parent)
            if mid not in p.children:
                p.children.append(mid)
        if not shared:
            for name, child in module._modules.items():
                if child is not None:
                    visit(child, f"{path}.{name}" if path else name, mid, name)
        return mid
    visit(model, "")
    # Include nested alias paths for export's module-stack lookup.
    def alias_paths(module, path, ancestors):
        paths[path] = identities[id(module)]
        if id(module) in ancestors:
            return
        ancestors = ancestors | {id(module)}
        for name, child in module._modules.items():
            if child is not None:
                alias_paths(child, f"{path}.{name}" if path else name, ancestors)
    alias_paths(model, "", set())
    return nodes, edges, identities, paths
