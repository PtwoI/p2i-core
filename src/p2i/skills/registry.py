"""Append-only local SQLite registry with source embedded in portable JSON records."""
import hashlib
import inspect
import json
import os
import re
import sqlite3
from pathlib import Path
from .schema import SkillSpec, SkillSearchResult, SkillError, HyperparameterSpec, RegisteredModuleImplementation, SkillOrigin, now
from .contracts import compatible, parameters_for
from .static import inspect_source, source_hash
from p2i.architecture.registry import registry as adapters

ELIGIBLE={'validated','evaluated','promoted'}
def fingerprint(spec):
    data={k:spec.model_dump(mode='json')[k] for k in ('implementation','input_contracts','output_contracts','hyperparameters','constraints')}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

def builtin_specs():
    for name,adapter in adapters.adapters.items():
        if not name.startswith('torch.nn.'):continue
        short=name.split('.')[-1]
        slug=re.sub(r'(?<!^)(?=[A-Z][a-z])','_',short).lower()
        fields=[]
        signature=inspect.signature(adapter.module_class)
        for key in getattr(adapter,'fields',[]):
            parameter=signature.parameters.get(key)
            default=parameter.default if parameter and parameter.default is not inspect.Parameter.empty else None
            required=parameter is not None and parameter.default is inspect.Parameter.empty
            kind='bool' if isinstance(default,bool) else 'float' if isinstance(default,float) else 'int' if isinstance(default,int) else 'str' if isinstance(default,str) else 'any'
            if key in {'in_features','out_features','in_channels','out_channels','num_heads','embed_dim','embedding_dim','num_embeddings','groups','num_features'}:kind='int'
            fields.append(HyperparameterSpec(name=key,type=kind,default=default,required=required,minimum=0 if key in ('p','dropout') else None,maximum=1 if key in ('p','dropout') else None))
        yield SkillSpec(id=f'builtin.{slug}',name=short,kind='canonical' if short in ('Embedding','MultiheadAttention') else 'primitive',status='promoted',description=f'PyTorch {short}; explicit trusted module adapter. Constructor and runtime validation apply.',semantic_tags=[short.lower(),'pytorch',*(['attention'] if short=='MultiheadAttention' else [])],hyperparameters=fields,implementation=RegisteredModuleImplementation(module_type=name),origin=SkillOrigin(kind='builtin'))

class SkillRegistry:
    def __init__(self,path=None,*,seed_builtins=True,allow_python=False):
        self.path=Path(path or os.environ.get('P2I_SKILL_REGISTRY',str(Path.home()/'.p2i'/'skills.sqlite3'))).expanduser()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.allow_python=allow_python
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS skills(id TEXT, revision INTEGER, body TEXT NOT NULL, PRIMARY KEY(id,revision))')
            db.execute('CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL)')
        if seed_builtins:
            for spec in builtin_specs():
                try:self.get(spec.id)
                except SkillError:self._append(spec,'builtin_seed')
    def _connect(self):return sqlite3.connect(self.path,timeout=15)
    def event(self,event,**data):
        body={'event':event,'timestamp':now().isoformat(),**data}
        with self._connect() as db:db.execute('INSERT INTO events(body) VALUES (?)',(json.dumps(body,sort_keys=True),))
        return body
    def events(self):
        with self._connect() as db:return [json.loads(r[0]) for r in db.execute('SELECT body FROM events ORDER BY seq')]
    def get(self,skill_id,revision=None):
        with self._connect() as db:
            row=db.execute('SELECT body FROM skills WHERE id=?'+(' AND revision=?' if revision is not None else '')+' ORDER BY revision DESC LIMIT 1',(skill_id,revision) if revision is not None else (skill_id,)).fetchone()
        if not row:raise SkillError('SKILL_NOT_FOUND',f'Unknown skill: {skill_id} revision {revision}')
        return SkillSpec.model_validate_json(row[0])
    def versions(self,skill_id):
        with self._connect() as db:return [SkillSpec.model_validate_json(r[0]) for r in db.execute('SELECT body FROM skills WHERE id=? ORDER BY revision',(skill_id,))]
    def list(self,*,status=None,kind=None):
        with self._connect() as db:rows=db.execute('SELECT s.body FROM skills s JOIN (SELECT id, MAX(revision) r FROM skills GROUP BY id) v ON s.id=v.id AND s.revision=v.r ORDER BY s.id').fetchall()
        specs=[SkillSpec.model_validate_json(r[0]) for r in rows]
        return [s for s in specs if (status is None or s.status in ([status] if isinstance(status,str) else status)) and (kind is None or s.kind==kind)]
    def _append(self,spec,event,expected=None):
        spec=spec.model_copy(deep=True)
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous=db.execute('SELECT MAX(revision) FROM skills WHERE id=?',(spec.id,)).fetchone()[0]
            if expected is not None and previous!=expected:raise SkillError('STALE_SKILL_REVISION','Skill changed during operation')
            spec.revision=0 if previous is None else previous+1;spec.updated_at=now()
            db.execute('INSERT INTO skills VALUES (?,?,?)',(spec.id,spec.revision,spec.model_dump_json()))
            db.execute('INSERT INTO events(body) VALUES (?)',(json.dumps({'event':event,'skill_id':spec.id,'revision':spec.revision,'timestamp':now().isoformat()}),))
        return spec
    def register(self,skill):
        spec=SkillSpec.model_validate(skill.model_dump() if isinstance(skill,SkillSpec) else skill)
        if spec.id.startswith('builtin.') or spec.origin.kind=='builtin':raise SkillError('RESERVED_ID','Built-in identity is reserved')
        if spec.implementation.type=='python_module':
            spec.implementation.sha256=inspect_source(spec.implementation.source,spec.implementation.class_name)
        existing=self.versions(spec.id)
        if existing and fingerprint(existing[-1])==fingerprint(spec) and all(getattr(existing[-1],k)==getattr(spec,k) for k in ('name','description','semantic_tags','kind')):return existing[-1]
        if existing:raise SkillError('SKILL_EXISTS','Use update() to create a new revision')
        spec.status='candidate';spec.validation=None;spec.evaluation=[]
        return self._append(spec,'skill_ingestion')
    def update(self,skill,*,reason='external revision'):
        spec=SkillSpec.model_validate(skill.model_dump() if isinstance(skill,SkillSpec) else skill)
        old=self.get(spec.id)
        if old.origin.kind=='builtin':raise SkillError('RESERVED_ID','Derive a new skill rather than changing built-ins')
        if spec.implementation.type=='python_module':spec.implementation.sha256=inspect_source(spec.implementation.source,spec.implementation.class_name)
        if fingerprint(old)==fingerprint(spec) and all(getattr(old,k)==getattr(spec,k) for k in ('name','description','semantic_tags','kind')):return old
        spec.origin.parent_skill_id=old.id;spec.origin.parent_skill_revision=old.revision;spec.origin.reason=reason
        spec.status='candidate';spec.validation=None;spec.evaluation=[];spec.created_at=old.created_at
        return self._append(spec,'skill_revision',old.revision)
    def validate(self,skill_id,*,parameters=None,dimensions=None,timeout=30,seed=0):
        from .validation import validate_skill
        spec=self.get(skill_id)
        report=validate_skill(spec,parameters=parameters,dimensions=dimensions,timeout=timeout,seed=seed)
        if spec.origin.kind!='builtin':
            spec.validation=report;spec.status='validated' if report.valid else 'candidate'
            self._append(spec,'skill_validation',spec.revision)
        self.event('validation_outcome',skill_id=skill_id,report=report.model_dump(mode='json'))
        return report
    def promote(self,skill_id):
        spec=self.get(skill_id)
        if spec.status=='promoted':return spec
        if spec.status not in ('validated','evaluated') or not spec.validation or not spec.validation.valid or spec.validation.fingerprint!=fingerprint(spec):raise SkillError('VALIDATION_REQUIRED','Promotion requires current successful forward/backward/gradient/re-trace validation')
        spec.status='promoted';return self._append(spec,'skill_promotion',spec.revision)
    def deprecate(self,skill_id):return self._status(skill_id,'deprecated')
    def reject(self,skill_id):return self._status(skill_id,'rejected')
    def _status(self,skill_id,status):
        spec=self.get(skill_id);spec.status=status;return self._append(spec,'skill_'+status,spec.revision)
    def record_evaluation(self,skill_id,record):
        spec=self.get(skill_id)
        if spec.status not in ELIGIBLE:raise SkillError('VALIDATION_REQUIRED','Evaluate a validated skill first')
        spec.evaluation.append(record)
        if spec.status=='validated':spec.status='evaluated'
        return self._append(spec,'skill_evaluation',spec.revision)
    def search(self,query='',*,input_contract=None,output_contract=None,tags=None,status=None,parameters=None):
        from p2i.architecture.schema import TensorContract
        inputs=TensorContract.model_validate(input_contract) if isinstance(input_contract,dict) else input_contract
        outputs=TensorContract.model_validate(output_contract) if isinstance(output_contract,dict) else output_contract
        words=set(re.findall(r'\w+',query.casefold()));results=[]
        for s in self.list(status=status if status is not None else sorted(ELIGIBLE)):
            hay=set(re.findall(r'\w+',' '.join([s.id,s.name,s.description,*s.semantic_tags]).casefold()))
            text=len(words & hay)/max(1,len(words))
            if words and not text:continue
            if tags and not set(tags)<=set(s.semantic_tags):continue
            bindings=dict(parameters or {});known=True
            ok=True
            for wanted,contracts in ((inputs,s.input_contracts),(outputs,s.output_contracts)):
                if wanted:
                    if not contracts:known=False
                    elif len(contracts)!=1 or not compatible(contracts[0],wanted,bindings):ok=False
            if not ok:continue
            if parameters:
                try:parameters_for(s,parameters)
                except SkillError:continue
            contract=1.0 if known and (inputs or outputs) else 0.0
            status_score=1.0 if s.status=='promoted' else .5 if s.status in ELIGIBLE else 0
            reasons=['Contract compatible' if contract else 'Contract not proven; parameterized preview required',f'Status: {s.status}',f'{len(words & hay)} text token matches']
            results.append(SkillSearchResult(skill_id=s.id,revision=s.revision,name=s.name,score=4*text+2*contract+status_score,text_score=text,contract_score=contract,status_score=status_score,compatible=known,reasons=reasons))
        results.sort(key=lambda r:(-r.score,r.skill_id))
        self.event('skill_search',query=query,results=[r.skill_id for r in results])
        return results
    def export(self,path):
        specs=[v.model_dump(mode='json') for s in self.list() for v in self.versions(s.id)]
        Path(path).write_text(json.dumps({'version':'0.1','skills':specs,'events':self.events()},indent=2,sort_keys=True)+'\n')
    def import_file(self,path):
        data=json.loads(Path(path).read_text())
        if data.get('version')!='0.1':raise SkillError('SCHEMA_ERROR','Unsupported registry export version')
        result=[]
        # Imported external records always require local validation; embedded reports are evidence, not trust.
        for raw in data['skills']:
            s=SkillSpec.model_validate(raw)
            if s.origin.kind=='builtin':continue
            result.append(self.update(s,reason='registry import') if self.versions(s.id) else self.register(s))
        return result
