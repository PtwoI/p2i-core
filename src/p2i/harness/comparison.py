"""Observed snapshot alignment; never equates transient tensor IDs between traces."""
from collections import Counter


def operation_keys(ir):
    paths={m['id']:m['qualified_name'] for m in ir['modules']}
    occurrences=Counter();calls={}
    for call in sorted((ir.get('runtime') or {}).get('calls',[]),key=lambda c:c['sequence']):
        path=paths.get(call['module_id'],'?');occurrences[path]+=1
        calls[call['id']]=(path,occurrences[path])
    counts=Counter();result={}
    for op in ir['operations']:
        if op['graph']!='runtime':continue
        parent=calls.get(op.get('call_id'),(paths.get(op.get('parent_module_id'),'?'),0))
        base=(*parent,op['op_type']);counts[base]+=1
        result[op['id']]=repr((*base,counts[base]))
    return result


def compare_observations(before,after):
    a,b=before['model'],after['model'];ak,bk=operation_keys(a),operation_keys(b)
    def paths(ir,keys):return sorted({(keys[e['source_id']],keys[e['target_id']]) for e in ir['data_edges'] if e['source_id'] in keys and e['target_id'] in keys})
    ap,bp=set(paths(a,ak)),set(paths(b,bk))
    am={m['qualified_name']:m for m in a['modules']};bm={m['qualified_name']:m for m in b['modules']}
    changed=[]
    for path in sorted(am.keys()|bm.keys()):
        old,new=am.get(path),bm.get(path)
        ac=before['configuration'].get(old['id'],{}) if old else None
        bc=after['configuration'].get(new['id'],{}) if new else None
        if ac!=bc or old is None or new is None:changed.append({'path':path,'before':ac,'after':bc})
    def outputs(ir,keys,inspection):
        tensors={t['id']:t for t in ir['tensors']};result={}
        for op in ir['operations']:
            if op['id'] not in keys:continue
            for slot,tid in enumerate(op['output_tensor_ids']):
                t=tensors[tid];capture=inspection.get('tensors',{}).get(tid,{})
                result[(keys[op['id']],slot)]={
                    'shape':t['shape'],'dtype':t['dtype'],
                    'grid':capture.get('grid'),
                    'sample':capture.get('values'),  # Older bounded sidecars.
                }
        return result
    ao=outputs(a,ak,before['inspection']);bo=outputs(b,bk,after['inspection'])
    tensors=[{'key':str(k),'before':ao[k],'after':bo[k]} for k in sorted(ao.keys()&bo.keys()) if ao[k]!=bo[k]]
    return {'alignment':'module path / call occurrence / operator type / occurrence; structural matching, not semantic equivalence','before_keys':ak,'after_keys':bk,'modules':changed,'tensors':tensors,'operations':{'before':len(ak),'after':len(bk),'added':sorted(set(bk.values())-set(ak.values())),'removed':sorted(set(ak.values())-set(bk.values()))},'paths':{'added':sorted(bp-ap),'removed':sorted(ap-bp)},'parameters':{'before':a['modules'][0]['parameter_count'],'after':b['modules'][0]['parameter_count']}}
