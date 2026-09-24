from copy import deepcopy
from p2i.architecture.schema import ArchitectureIR, ArchitectureNode, Registration, TensorContract
from .schema import SkillSpec, SkillOrigin, CompositeIRImplementation, SkillError

def skill_from_subgraph(architecture_ir,*,node_ids,name,skill_id,input_contracts,output_contracts):
    """Explicit consecutive Sequential region (or one complete module subtree)."""
    arch=architecture_ir;byid={n.id:n for n in arch.nodes}
    ids=[next((n.id for n in arch.nodes if n.id==i or n.qualified_name==i),i) for i in node_ids]
    if not ids or len(set(ids))!=len(ids) or any(i not in byid for i in ids):raise SkillError('INVALID_BOUNDARY','Select existing distinct nodes')
    if not input_contracts or not output_contracts:raise SkillError('INVALID_BOUNDARY','Explicit input and output contracts required')
    if len(ids)>1:
        parents=[n for n in arch.nodes if n.module_type=='torch.nn.Sequential' and all(i in [r.node_id for r in n.children] for i in ids)]
        if len(parents)!=1:raise SkillError('INVALID_BOUNDARY','Multiple nodes must be consecutive children of one Sequential')
        siblings=[r.node_id for r in parents[0].children];start=siblings.index(ids[0])
        if siblings[start:start+len(ids)]!=ids:raise SkillError('INVALID_BOUNDARY','Boundary must preserve consecutive Sequential order')
    selected={}
    def visit(i):
        if i in selected:return
        n=deepcopy(byid[i])
        if not n.editable or n.managed_by:raise SkillError('UNSUPPORTED_EXTRACTION','Only reconstructable adapter subtrees can be exported')
        if n.module_type.startswith('p2i.skill.'):raise SkillError('UNSUPPORTED_EXTRACTION','Nested skill dependencies are not portable in Phase 3; extract adapter nodes')
        selected[i]=n
        for c in n.children:visit(c.node_id)
    for i in ids:visit(i)
    root=ids[0]
    if len(ids)>1:
        root='fragment-root'
        selected[root]=ArchitectureNode(id=root,name=name,qualified_name='',kind='container',module_type='torch.nn.Sequential',editable=True,children=[Registration(name=str(i),node_id=n) for i,n in enumerate(ids)])
    def paths(i,path):
        n=selected[i];n.qualified_name=path;n.source_observed_module_id=None;n.latest_observed_module_id=None
        for c in n.children:paths(c.node_id,f'{path}.{c.name}' if path else c.name)
    paths(root,'')
    fragment=ArchitectureIR(model_name=name,root_id=root,nodes=list(selected.values()),seed=arch.seed)
    return SkillSpec(id=skill_id,name=name,kind='composite',description=f'Explicit extracted component: {name}',input_contracts=input_contracts,output_contracts=output_contracts,implementation=CompositeIRImplementation(architecture=fragment),origin=SkillOrigin(kind='model-extracted',source_model_id=arch.model_name,source_revision=arch.revision,source_nodes=ids))

def extract_skill(harness,*,node_id,name,skill_id,input_contracts=None,output_contracts=None):
    node=harness._resolve(harness.architecture(),node_id)
    return skill_from_subgraph(harness.architecture(),node_ids=[node.id],name=name,skill_id=skill_id,input_contracts=input_contracts or node.input_contracts,output_contracts=output_contracts or node.output_contracts)
