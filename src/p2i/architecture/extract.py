from p2i.analyze.modules import extract as extract_modules
from .schema import ArchitectureIR,ArchitectureNode,Registration,TensorContract
from .registry import registry as default_registry

def extract_architecture(model,observed=None,registry=None,seed=0):
    registry=registry or default_registry
    modules,edges,identities,paths=extract_modules(model)
    originals={f'a{i[1:]}':module for module in model.modules() for i in [identities[id(module)]]}
    observed_by_path={m.qualified_name:m for m in observed.modules} if observed else {}
    tensors={t.id:t for t in observed.tensors} if observed else {}
    nodes=[]
    def contracts(ids):return [TensorContract(shape=tensors[i].shape,dtype=tensors[i].dtype) for i in ids if i in tensors]
    for m in modules:
        aid='a'+m.id[1:];module=originals[aid];adapter=registry.find(module);obs=observed_by_path.get(m.qualified_name)
        nodes.append(ArchitectureNode(id=aid,name=m.name,qualified_name=m.qualified_name,
            kind='container' if adapter and adapter.container else 'module' if adapter else 'custom',
            module_type=adapter.module_type if adapter else m.module_path+'.'+m.class_name,
            parameters=adapter.extract(module) if adapter else {},
            children=[Registration(name=e.name,node_id='a'+e.child_id[1:]) for e in edges if e.parent_id==m.id],
            source_observed_module_id=obs.id if obs else None,latest_observed_module_id=obs.id if obs else None,
            editable=adapter is not None,edit_restrictions=[] if adapter else ['No reconstruction adapter is available. Constructor is read-only; live sessions retain the original forward implementation.'],
            input_contracts=contracts(obs.input_tensor_ids) if obs else [],output_contracts=contracts(obs.output_tensor_ids) if obs else [],contracts_revision=0 if obs else None,training=module.training))
    byid={n.id:n for n in nodes}
    def managed(nid,owner):
        node=byid[nid];node.managed_by=owner;node.editable=False
        node.edit_restrictions.append(f'Internal module managed by adapter {owner}; edit its parent instead.')
        for r in node.children:managed(r.node_id,owner)
    for n in nodes:
        adapter=registry.adapters.get(n.module_type)
        if adapter and adapter.owns_children:
            for r in n.children:managed(r.node_id,n.id)
    return ArchitectureIR(model_name=type(model).__name__,root_id='a0',nodes=nodes,seed=seed),originals
