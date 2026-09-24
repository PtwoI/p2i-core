import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from .schema import SkillValidationReport, SkillError
from .static import inspect_source
from .contracts import parameters_for
from .registry import fingerprint
from p2i.architecture.schema import ValidationIssue

def validate_skill(spec,*,parameters=None,dimensions=None,timeout=30,seed=0):
    if not 0<timeout<=300:raise SkillError('INVALID_PARAMETER','timeout must be in (0, 300] seconds')
    report=SkillValidationReport(fingerprint=fingerprint(spec))
    try:
        if spec.implementation.type=='python_module':inspect_source(spec.implementation.source,spec.implementation.class_name)
        report.static_valid=True
        values=parameters_for(spec,parameters or {});report.parameters=values
    except (SkillError,ValueError) as exc:
        report.issues.append(ValidationIssue(category='static',message=str(exc),exception_type=getattr(exc,'code',type(exc).__name__)));return report
    dims={'B':2,'T':7,'D':8,'C':3,'H':8,'W':8,**(dimensions or {})}
    cases=[dims]
    if any('T' in c.shape for c in spec.input_contracts):cases.append({**dims,'T':dims['T']+4})
    with tempfile.TemporaryDirectory(prefix='p2i-skill-') as tmp:
        request=Path(tmp)/'request.json';result=Path(tmp)/'result.json';progress=Path(tmp)/'stage'
        request.write_text(json.dumps({'skill':spec.model_dump(mode='json'),'parameters':values,'cases':cases,'seed':seed,'fingerprint':report.fingerprint,'cpu_seconds':max(1,math.ceil(timeout))}))
        env={k:v for k,v in os.environ.items() if k in ('PATH','SYSTEMROOT','LD_LIBRARY_PATH')}
        # Explicit package search paths, but no API keys, proxy credentials or arbitrary environment.
        env.update(PYTHONPATH=os.pathsep.join(p for p in sys.path if p),HOME=tmp,TMPDIR=tmp,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        try:
            process=subprocess.run([sys.executable,'-m','p2i.skills.worker',str(request),str(result),str(progress)],cwd=tmp,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=timeout)
            if result.exists():return SkillValidationReport.model_validate_json(result.read_text())
            report.issues.append(ValidationIssue(category=progress.read_text() if progress.exists() else 'import',message=f'Validation worker exited without report ({process.returncode})',exception_type='WORKER_FAILED'))
        except subprocess.TimeoutExpired:
            report.issues.append(ValidationIssue(category=progress.read_text() if progress.exists() else 'import',message=f'Validation exceeded {timeout}s wall-clock budget',exception_type='TIMEOUT'))
    return report
