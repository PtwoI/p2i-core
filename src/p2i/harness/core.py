from copy import deepcopy
from threading import RLock
import torch
from p2i.actions import parse_action,ActionResult,Batch
from p2i.architecture import extract_architecture
from p2i.architecture.schema import ArchitectureIR,ArchitectureNode,Registration,ConstructorSpec,TensorContract,architecture_diff,ValidationReport
from p2i.architecture.registry import registry as default_registry
from p2i.architecture.builder import build_architecture
from .validation import validate_architecture

from p2i.skills.harness import SkillHarnessMixin

class Harness(SkillHarnessMixin):
    """Deterministic architecture transactions with in-memory original state.

    apply/preview validate construction and, when examples exist, forward execution.
    No tracing occurs until retrace(). JSON artifacts never load Python source.
    """
    def __init__(self,model,ir=None,example_inputs=None,*,example_args=None,example_kwargs=None,registry=None,seed=0,skill_registry=None):
        if example_inputs is not None and example_args is not None:raise ValueError('Use example_inputs or example_args, not both')
        args=example_args if example_args is not None else example_inputs
        if args is not None and not isinstance(args,tuple):raise TypeError('Example positional inputs must be a tuple')
        self.registry=(registry or default_registry).copy()
        self._skill_registry=skill_registry
        self._model,self._args,self._kwargs=deepcopy((model,args,example_kwargs or {}))
        self._arch,self._originals=extract_architecture(self._model,ir,self.registry,seed)
        self._init_session(ir)
    def _init_session(self,ir):
        self._experiments=[];self.last_evaluation=None
        self._lock=RLock();self.latest_ir=deepcopy(ir);self.observed_revision=0 if ir else None
        self._observed_bindings={n.id:n.latest_observed_module_id for n in self._arch.nodes if n.latest_observed_module_id}
        self._source_bindings={n.id:n.source_observed_module_id for n in self._arch.nodes if n.source_observed_module_id}
        self.observed_configuration={}
        if ir and self._model is not None:self._record_configuration(self._model,ir)
        self.last_validation=None;self.last_weights=None;self.last_comparison=None;self.inspection=None
        self.previous_observation=None
        self._revisions={0:self._arch.model_copy(deep=True)};self._history=[];self._undo=[];self._redo=[];self._action_counter=0
    def _record_configuration(self,model,ir):
        observed={m.qualified_name:m.id for m in ir.modules};result={}
        for path,module in model.named_modules():
            adapter=self.registry.find(module)
            if adapter and path in observed:result[observed[path]]={'module_type':adapter.module_type,'parameters':adapter.extract(module)}
        self.observed_configuration=result
    @classmethod
    def from_architecture(cls,architecture,*,example_inputs=None,example_kwargs=None,registry=None,skill_registry=None):
        obj=cls.__new__(cls);obj.registry=(registry or default_registry).copy();obj._skill_registry=skill_registry
        obj._arch=ArchitectureIR.model_validate(architecture.model_dump() if isinstance(architecture,ArchitectureIR) else architecture)
        obj._originals={};obj._model=None;obj._args=example_inputs;obj._kwargs=example_kwargs or {}
        obj._init_session(None);obj._revisions={obj._arch.revision:obj._arch.model_copy(deep=True)}
        from p2i.skills.application import ensure_adapter
        for node in obj._arch.nodes:
            if node.skill_id:
                spec=obj.skill_registry.get(node.skill_id,node.skill_revision)
                ensure_adapter(obj,spec)
        return obj
    def architecture(self):
        with self._lock:return self._arch.model_copy(deep=True)
    def _resolve(self,arch,target):
        for n in arch.nodes:
            if n.id==target or n.qualified_name==target:return n
        raise ValueError(f'Unknown architecture target: {target}')
    def _fresh(self,arch):
        i=0;ids={n.id for n in arch.nodes}|set(self._originals)
        for past in self._revisions.values():ids.update(n.id for n in past.nodes)
        while f'a{i}' in ids:i+=1
        return f'a{i}'
    def _new(self,arch,spec):
        adapter=self.registry.get(spec.module_type);adapter.validate(spec.parameters)
        # Canonicalize defaults through the explicit adapter.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []):
            torch.manual_seed(arch.seed);module=adapter.build(spec.parameters)
        node=ArchitectureNode(id=self._fresh(arch),name='new',qualified_name='',kind='container' if adapter.container else 'module',module_type=spec.module_type,parameters=adapter.extract(module),editable=True,training=arch.nodes[0].training)
        if adapter.owns_children and list(module.children()):
            # Internals are rebuilt by the adapter; the next trace exposes them.
            node.edit_restrictions=['Adapter owns its internal registered modules.']
        arch.nodes.append(node);return node
    def _normalize(self,arch):
        byid={n.id:n for n in arch.nodes};seen=set()
        def walk(nid,path):
            if nid in seen:return
            seen.add(nid);n=byid[nid];n.qualified_name=path;n.name=path.rsplit('.',1)[-1] if path else arch.model_name
            if n.module_type=='torch.nn.ModuleList':
                for i,r in enumerate(n.children):r.name=str(i)
            for r in n.children:walk(r.node_id,f'{path}.{r.name}' if path else r.name)
        walk(arch.root_id,'');arch.nodes=[n for n in arch.nodes if n.id in seen]
    def _edit(self,arch,action):
        if action.expected_revision is not None and action.expected_revision!=self._arch.revision:raise ValueError(f'Stale revision: expected {action.expected_revision}, current {self._arch.revision}')
        if isinstance(action,Batch):
            for a in action.actions:self._edit(arch,a)
            return
        if action.type in ('replace_with_skill','insert_skill'):
            from p2i.skills.application import resolve_action
            from p2i.actions import ReplaceModule,InsertModule
            spec,constructor=resolve_action(self,arch,action)
            previous={n.id for n in arch.nodes}
            if action.type=='replace_with_skill':
                node=self._resolve(arch,action.target)
                self._edit(arch,ReplaceModule(target=node.id,replacement=constructor))
            else:
                self._edit(arch,InsertModule(parent=action.parent,index=action.index,module=constructor))
                node=next(n for n in arch.nodes if n.id not in previous)
            node.skill_id=spec.id;node.skill_revision=spec.revision
            return
        if action.type=='insert_module':
            parent=self._resolve(arch,action.parent)
            if parent.module_type not in ('torch.nn.Sequential','torch.nn.ModuleList') or parent.managed_by:raise ValueError('Insertion requires an editable Sequential or ModuleList container')
            if not 0<=action.index<=len(parent.children):raise ValueError('Insertion index is out of range')
            new=self._new(arch,action.module)
            name=f'p2i_{new.id}'
            parent.children.insert(action.index,Registration(name=name,node_id=new.id));return
        node=self._resolve(arch,action.target)
        if node.managed_by:raise ValueError(f'Module is owned by adapter {node.managed_by}; edit that parent instead')
        if action.type=='set_parameter':
            if not node.editable:raise ValueError('Read-only constructor: no reconstruction adapter is available')
            if action.parameter not in node.parameters:raise ValueError(f'Unknown constructor parameter {action.parameter}')
            if node.skill_id and node.module_type.startswith('p2i.skill.'):
                raise ValueError('Reapply the skill with a validated configuration instead of mutating its pinned constructor')
            node.parameters[action.parameter]=action.value
            if node.skill_id:node.skill_id=None;node.skill_revision=None
        elif action.type=='replace_module':
            new=self._new(arch,action.replacement)
            node.module_type=new.module_type;node.parameters=new.parameters;node.kind=new.kind;node.children=[];node.editable=True;node.edit_restrictions=new.edit_restrictions
            node.skill_id=None;node.skill_revision=None
            arch.nodes.remove(new)
        elif action.type=='remove_module':
            if node.id==arch.root_id:raise ValueError('Cannot remove the root')
            parents=[p for p in arch.nodes if any(r.node_id==node.id for r in p.children)]
            if any(p.module_type not in ('torch.nn.Sequential','torch.nn.ModuleList') or p.managed_by for p in parents):raise ValueError('Removal is only supported inside Sequential/ModuleList')
            for p in parents:p.children=[r for r in p.children if r.node_id!=node.id]
        elif action.type=='wrap_module':
            new=self._new(arch,action.module)
            wrapper=self._new(arch,ConstructorSpec(module_type='torch.nn.Sequential'))
            # Redirect all registrations to preserve shared instance identity.
            for p in arch.nodes:
                if p.id in (wrapper.id,new.id):continue
                for r in p.children:
                    if r.node_id==node.id:r.node_id=wrapper.id
            if arch.root_id==node.id:arch.root_id=wrapper.id
            order=[node.id,new.id] if action.position=='after' else [new.id,node.id]
            wrapper.children=[Registration(name=str(i),node_id=nid) for i,nid in enumerate(order)]
    def _propose(self,action):
        parsed=parse_action.validate_python(action.model_dump() if hasattr(action,'model_dump') else action)
        candidate=self._arch.model_copy(deep=True)
        self._edit(candidate,parsed);self._normalize(candidate)
        candidate=ArchitectureIR.model_validate(candidate.model_dump())
        report,_,weights,_=validate_architecture(candidate,originals=self._originals,registry=self.registry,args=self._args,kwargs=self._kwargs)
        return parsed,candidate,report,weights
    def _action(self,action,commit):
        with self._lock:
            self._action_counter+=1
            kind=getattr(action,'type',None) or (action.get('type','unknown') if isinstance(action,dict) else 'unknown')
            result=ActionResult(success=False,action_id=f'action-{self._action_counter}',action_type=kind,revision=self._arch.revision)
            try:
                parsed,candidate,report,weights=self._propose(action)
                result.validation=report;result.architecture_valid=report.schema_valid and report.structural_valid and report.constructor_valid
                result.errors=[i.message for i in report.issues if i.severity=='error']
                result.warnings=[i.message for i in report.issues if i.severity=='warning']
                result.diff=architecture_diff(self._arch,candidate)
                result.changed_nodes=list(dict.fromkeys(d.node_id for d in result.diff.changes))
                if not report.valid:return result
                result.success=True
                if weights and weights.reinitialized:result.warnings.append(f'{len(weights.reinitialized)} state entries will be deterministically reinitialized; inspect weight_transfer after build.')
                if not result.changed_nodes:result.warnings.append('No architecture changes');return result
                if commit:
                    self._undo.append(self._arch.model_copy(deep=True));self._redo.clear()
                    self._commit(candidate,parsed.model_dump(mode='json'))
                    self.last_validation=report.model_copy(update={'revision':self._arch.revision})
                    result.revision=self._arch.revision;result.committed=True;result.validation=self.last_validation
                    result.diff=self._history[-1]['diff']
                return result
            except Exception as exc:
                result.errors=[str(exc)];result.error_code=getattr(exc,'code','INVALID_ACTION');result.architecture_valid=False;return result
    def preview(self,action):return self._action(action,False)
    def apply(self,action):
        result=self._action(action,True)
        self._event('architecture_action',result=result.model_dump(mode='json'))
        if result.committed:
            for nid in result.changed_nodes:
                node=next((n for n in self._arch.nodes if n.id==nid),None)
                if node and node.skill_id:self._event('skill_reuse',skill_id=node.skill_id,skill_revision=node.skill_revision,node_id=nid)
        return result
    def _commit(self,arch,action):
        for node in arch.nodes:
            node.latest_observed_module_id=self._observed_bindings.get(node.id)
            node.source_observed_module_id=node.source_observed_module_id or self._source_bindings.get(node.id)
        arch.revision=self._arch.revision+1;diff=architecture_diff(self._arch,arch)
        self._history.append({'revision':arch.revision,'action':action,'diff':diff})
        self._arch=arch;self._revisions[arch.revision]=arch.model_copy(deep=True)
        self.last_validation=None;self.last_weights=None;self.last_comparison=None
    def _restore(self,redo=False,expected_revision=None):
        with self._lock:
            if expected_revision is not None and expected_revision!=self._arch.revision:return {'success':False,'errors':['Stale revision'],'revision':self._arch.revision}
            source,dest=(self._redo,self._undo) if redo else (self._undo,self._redo)
            if not source:return {'success':False,'errors':['No revision to restore'],'revision':self._arch.revision}
            dest.append(self._arch.model_copy(deep=True));self._commit(source.pop(),{'type':'redo' if redo else 'undo'})
            return {'success':True,'revision':self._arch.revision}
    def undo(self,expected_revision=None):return self._restore(False,expected_revision)
    def redo(self,expected_revision=None):return self._restore(True,expected_revision)
    def history(self):
        with self._lock:return [{**deepcopy(e),'diff':e['diff'].model_dump(mode='json')} for e in self._history]
    def diff(self,from_revision=None,to_revision=None):
        with self._lock:
            to_revision=self._arch.revision if to_revision is None else to_revision
            from_revision=max(min(self._revisions),to_revision-1) if from_revision is None else from_revision
            return architecture_diff(self._revisions[from_revision],self._revisions[to_revision])
    def validate(self,*,backward=False):
        with self._lock:
            self.last_validation,_,_,_=validate_architecture(self._arch,originals=self._originals,registry=self.registry,args=self._args,kwargs=self._kwargs,backward=backward)
            self._event('validation_outcome',report=self.last_validation.model_dump(mode='json'))
            return self.last_validation.model_copy(deep=True)
    def build(self):
        with self._lock:
            model,self.last_weights,_=build_architecture(self._arch,originals=self._originals,registry=self.registry)
            return model
    def retrace(self,*,capture_values=False,**trace_options):
        from p2i import trace
        with self._lock:
            if self._args is None:raise ValueError('Re-trace requires example_inputs (use () for a model with no inputs)')
            report,model,weights,paths=validate_architecture(self._arch,originals=self._originals,registry=self.registry,args=self._args,kwargs=self._kwargs)
            self.last_validation=report
            if not report.valid or not report.forward_valid:raise ValueError('Build & re-trace failed validation: '+'; '.join(i.message for i in report.issues))
            from p2i.analyze.inspection import RuntimeInspection
            inspection=RuntimeInspection() if capture_values else None
            observed=trace(model,example_inputs=self._args,example_kwargs=self._kwargs,inspection=inspection,**trace_options)
            if any(a.status!='success' for a in observed.metadata.analysis if a.technique in ('runtime_hooks','runtime_dispatch')):raise ValueError('Re-trace did not complete runtime capture; latest observation retained')
            old=self.latest_ir
            if old:
                self.previous_observation={'model':old.model_dump(mode='json'),'revision':self.observed_revision,'configuration':deepcopy(self.observed_configuration),'inspection':self.inspection.as_dict() if self.inspection else {'tensors':{},'operations':{}}}
            self.last_comparison=self._compare(old,observed) if old else None
            self.inspection=inspection
            self._record_configuration(model,observed)
            self.latest_ir=observed;self.observed_revision=self._arch.revision;self.last_weights=weights
            bypath={m.qualified_name:m for m in observed.modules}
            tensor_by_id={t.id:t for t in observed.tensors}
            for n in self._arch.nodes:
                match=bypath.get(n.qualified_name);n.latest_observed_module_id=match.id if match else None
                if match:
                    n.input_contracts=[TensorContract(shape=tensor_by_id[t].shape,dtype=tensor_by_id[t].dtype) for t in match.input_tensor_ids]
                    n.output_contracts=[TensorContract(shape=tensor_by_id[t].shape,dtype=tensor_by_id[t].dtype) for t in match.output_tensor_ids]
                    n.contracts_revision=self._arch.revision
                if self._arch.revision==0 and match:n.source_observed_module_id=n.source_observed_module_id or match.id
            self._observed_bindings={n.id:n.latest_observed_module_id for n in self._arch.nodes if n.latest_observed_module_id}
            self._source_bindings.update({n.id:n.source_observed_module_id for n in self._arch.nodes if n.source_observed_module_id})
            self._event('retrace_result',modules=len(observed.modules),operations=len(observed.operations))
            return observed.model_copy(deep=True)
    @staticmethod
    def _compare(a,b):
        def counts(ir):return {'parameters':ir.modules[0].parameter_count,'modules':len(ir.modules),'operations':len(ir.operations),'runtime_operations':sum(o.graph=='runtime' for o in ir.operations)}
        def shapes(ir):
            ts={t.id:t.shape for t in ir.tensors}
            return {m.qualified_name:{'inputs':[ts[t] for t in m.input_tensor_ids],'outputs':[ts[t] for t in m.output_tensor_ids]} for m in ir.modules}
        x,y=shapes(a),shapes(b)
        return {'before':counts(a),'after':counts(b),'shape_changes':[{'path':p,'before':x.get(p),'after':y.get(p)} for p in sorted(x.keys()|y.keys()) if x.get(p)!=y.get(p)]}
    def observe(self,level='summary',node_id=None):
        with self._lock:
            if node_id is not None:return self._resolve(self._arch,node_id).model_dump(mode='json')
            if level=='architecture':return self.architecture().model_dump(mode='json')
            if level!='summary':raise ValueError('level must be summary or architecture')
            return {'model_name':self._arch.model_name,'revision':self._arch.revision,'observed_revision':self.observed_revision,'node_count':len(self._arch.nodes),'editable_node_count':sum(n.editable for n in self._arch.nodes),'observed_parameter_count':self.latest_ir.modules[0].parameter_count if self.latest_ir else None,'validation':self.last_validation.model_dump(mode='json') if self.last_validation else None,'recent_edits':self.history()[-5:],'weight_transfer':self.last_weights.model_dump(mode='json') if self.last_weights else None,'comparison':self.last_comparison}
    def provenance(self):
        with self._lock:
            return [{"architecture_node_id":n.id,"built_path":n.qualified_name,"source_observed_module_id":n.source_observed_module_id,"latest_observed_module_id":n.latest_observed_module_id,"contracts_revision":n.contracts_revision} for n in self._arch.nodes]
    def observe_json(self,**kwargs):
        import json
        return json.dumps(self.observe(**kwargs),sort_keys=True)
    def validation_json(self):return self.validate().model_dump_json()
    def diff_json(self,**kwargs):return self.diff(**kwargs).model_dump_json()
