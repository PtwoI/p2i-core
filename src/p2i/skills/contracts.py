"""Conservative shape unification and parameter validation; never evaluate expressions."""
from p2i.architecture.schema import TensorContract
from .schema import SkillError

def compatible(expected, actual, bindings=None):
    if bindings is None: bindings={}
    if expected.rank is not None and actual.rank is not None and expected.rank!=actual.rank:return False
    if expected.device and actual.device and expected.device!=actual.device:return False
    if expected.dtype and actual.dtype and expected.dtype!=actual.dtype:return False
    def family(c):
        if c.dtype_family:return c.dtype_family
        if c.dtype:
            if 'float' in c.dtype or 'bfloat' in c.dtype:return 'floating'
            if 'int' in c.dtype:return 'integer'
            if 'bool' in c.dtype:return 'boolean'
        return None
    if family(expected) and family(actual) and family(expected)!=family(actual):return False
    if expected.shape and actual.shape:
        if len(expected.shape)!=len(actual.shape):return False
        for e,a in zip(expected.shape,actual.shape):
            if isinstance(e,int) and isinstance(a,int) and e!=a:return False
            if isinstance(e,str) and isinstance(a,int):
                if e in bindings and bindings[e]!=a:return False
                bindings[e]=a
    return True

def check_contracts(spec, inputs, outputs, parameters):
    bindings=dict(parameters)
    for expected,actual in ((spec.input_contracts,inputs),(spec.output_contracts,outputs)):
        if expected and actual:
            if len(expected)!=len(actual) or any(not compatible(e,a,bindings) for e,a in zip(expected,actual)):
                raise SkillError('CONTRACT_MISMATCH','Skill tensor contracts do not match target',{'expected':[x.model_dump() for x in expected],'actual':[x.model_dump() for x in actual]})
    return bindings

def parameters_for(spec, supplied):
    fields={h.name:h for h in spec.hyperparameters}
    if set(supplied)-set(fields):raise SkillError('INVALID_PARAMETER',f'Unknown skill parameters: {sorted(set(supplied)-set(fields))}')
    values={h.name:h.default for h in fields.values() if h.default is not None}
    values.update(supplied)
    for name,h in fields.items():
        if h.required and name not in values:raise SkillError('INVALID_PARAMETER',f'Missing required parameter: {name}')
        if name not in values:continue
        value=values[name]
        ok={'int':type(value) is int,'float':type(value) in (int,float),'bool':type(value) is bool,'str':isinstance(value,str),'array':isinstance(value,list),'any':True}[h.type]
        if not ok:raise SkillError('INVALID_PARAMETER',f'{name} must have type {h.type}')
        if h.minimum is not None and value<h.minimum or h.maximum is not None and value>h.maximum:raise SkillError('INVALID_PARAMETER',f'{name} is outside allowed bounds')
        if h.choices and value not in h.choices:raise SkillError('INVALID_PARAMETER',f'{name} must be one of {h.choices}')
    for c in spec.constraints:
        left=values.get(c.left);right=values.get(c.right) if isinstance(c.right,str) else c.right
        if c.kind=='divisible' and (not isinstance(left,int) or not isinstance(right,int) or right<=0 or left%right):raise SkillError('INVALID_PARAMETER',f'{c.left} must be divisible by {c.right}')
        if c.kind=='equal' and left!=right:raise SkillError('INVALID_PARAMETER',f'{c.left} must equal {c.right}')
        if c.kind=='device' and c.right!='cpu':raise SkillError('DEVICE_UNSUPPORTED','Phase 3 candidate validation supports CPU only')
    return values
