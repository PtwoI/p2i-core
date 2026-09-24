"""A deliberately small source subset, not a security sandbox."""
import ast
import hashlib
from .schema import SkillError

def source_hash(source):return hashlib.sha256(source.encode()).hexdigest()

def inspect_source(source, class_name):
    try:tree=ast.parse(source)
    except SyntaxError as exc:raise SkillError('SYNTAX_ERROR',str(exc),{'line':exc.lineno}) from exc
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    if len(classes)!=1 or classes[0].name!=class_name:raise SkillError('STATIC_REJECTED','Require exactly one primary module class matching class_name')
    cls=classes[0]
    if len(cls.bases)!=1 or ast.unparse(cls.bases[0]) not in ('nn.Module','torch.nn.Module'):raise SkillError('STATIC_REJECTED','Primary class must inherit nn.Module or torch.nn.Module')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    if not {'__init__','forward'}<=methods.keys():raise SkillError('STATIC_REJECTED','Explicit __init__ and forward methods are required')
    if methods['forward'].args.vararg or methods['forward'].args.kwarg:raise SkillError('STATIC_REJECTED','Forward must use explicit arguments')
    for n in tree.body:
        if not isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef)) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)):
            raise SkillError('STATIC_REJECTED','Top level must contain imports, a docstring, and the primary class only')
    forbidden={'eval','exec','compile','open','getattr','setattr','delattr','globals','locals','vars','__import__','breakpoint','input','exit','quit','system','popen','spawn','fork','load','save','load_library','set_default_device','set_default_dtype','set_num_threads','manual_seed','set_rng_state','optim','backward','zero_grad','hub','distributed','multiprocessing','utils','cuda','ops','classes','serialization','package','jit','from_file','frombuffer','download_url_to_file'}
    imported=set()
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            imported.update(a.asname or a.name.split('.')[0] for a in n.names)
    for n in ast.walk(tree):
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            for target in targets:
                while isinstance(target,(ast.Attribute,ast.Subscript)):target=target.value
                if isinstance(target,ast.Name) and target.id in imported:
                    raise SkillError('STATIC_REJECTED','Mutation of imported namespaces is disallowed')
        if isinstance(n,ast.Import):names=[a.name for a in n.names]
        elif isinstance(n,ast.ImportFrom):names=[n.module or '']
        else:names=[]
        if any(x not in ('torch','torch.nn','torch.nn.functional','math') for x in names):raise SkillError('DEPENDENCY_NOT_ALLOWED','Only torch, torch.nn, torch.nn.functional and math imports are allowed',{'imports':names})
        if isinstance(n,ast.ImportFrom) and (n.level or any(a.name=='*' or a.name.startswith('_') or a.name in forbidden for a in n.names)):raise SkillError('STATIC_REJECTED','Relative, wildcard and private imports are disallowed')
        if isinstance(n,(ast.Global,ast.Nonlocal,ast.AsyncFunctionDef)):raise SkillError('STATIC_REJECTED','Global/nonlocal and asynchronous behavior are disallowed')
        if isinstance(n,ast.Name) and (n.id in forbidden or n.id.startswith('__') and n.id not in ('__init__',)):raise SkillError('STATIC_REJECTED',f'Disallowed name: {n.id}')
        if isinstance(n,ast.Attribute) and (n.attr in forbidden or n.attr.startswith('_') and n.attr!='__init__'):raise SkillError('STATIC_REJECTED',f'Disallowed attribute: {n.attr}')
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.decorator_list:raise SkillError('STATIC_REJECTED','Decorators are disallowed')
    for n in cls.body:
        if not isinstance(n,ast.FunctionDef) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant)):
            raise SkillError('STATIC_REJECTED','Class-level execution is disallowed')
    return source_hash(source)
