"""Opt-in bounded runtime sidecar. Never changes ModelIR's saved schema."""
import math
import torch

class RuntimeInspection:
    def __init__(self,*,max_tensors=64,max_elements=4096,sample_size=0,max_operations=256,
                 max_summary_elements=262144,grid_size=32):
        if not (0<=max_tensors<=256 and 0<=max_elements<=16384 and 0<=sample_size<=64 and
                0<=max_operations<=1024 and 0<=max_summary_elements<=1048576 and 1<=grid_size<=32):
            raise ValueError('Inspection limits exceed hard safety caps')
        self.max_tensors=max_tensors;self.max_elements=max_elements;self.sample_size=sample_size
        self.max_operations=max_operations;self.max_summary_elements=max_summary_elements;self.grid_size=grid_size
        self.tensors={};self.operations={}
    def tensor(self,info,tensor):
        if len(self.tensors)>=self.max_tensors:return
        entry={'elements':tensor.numel(),'values':None,'grid':None,
               'message':'Grid unavailable: size, layout or device threshold.'}
        self.tensors[info.id]=entry
        if (tensor.numel()==0 or tensor.numel()>self.max_summary_elements or tensor.device.type!='cpu'
                or tensor.layout!=torch.strided or tensor.is_complex() or tensor.is_quantized):return
        try:
            # A bounded, whole-tensor summary lives outside the Model IR. Disable dispatch
            # so visualization reductions cannot masquerade as model computation.
            with torch._C._DisableTorchDispatch(),torch.no_grad():
                value=tensor.detach().float()
                flat=value.flatten()
                clean=lambda v:v if math.isfinite(v) else None
                finite=torch.isfinite(flat)
                finite_values=flat[finite]
                if self.sample_size and tensor.numel()<=self.max_elements:
                    entry['values']=[clean(v) for v in flat[:self.sample_size].tolist()]
                if finite_values.numel():
                    entry.update(min=clean(finite_values.min().item()),max=clean(finite_values.max().item()),
                                 mean=clean(finite_values.mean().item()),
                                 std=clean(finite_values.std(unbiased=False).item()))
                # Keep the final two numeric axes. Higher axes are explicitly averaged;
                # one-dimensional and scalar tensors use a single-row grid.
                source_shape=list(tensor.shape)
                if value.ndim==0:plane=value.reshape(1,1)
                elif value.ndim==1:plane=value.reshape(1,-1)
                else:plane=value.reshape(-1,*value.shape[-2:])
                valid=torch.isfinite(plane)
                filled=torch.where(valid,plane,0)
                if value.ndim>=3:
                    filled=filled.mean(dim=0)
                    coverage=valid.float().mean(dim=0)
                else:
                    filled=filled.reshape(plane.shape[-2:])
                    coverage=valid.float().reshape(plane.shape[-2:])
                # Preserve every position while the final two axes fit in the
                # grid. Only larger planes need bounded regional means.
                scale=max(1,filled.shape[0]/self.grid_size,filled.shape[1]/self.grid_size)
                height=max(1,min(self.grid_size,round(filled.shape[0]/scale)))
                width=max(1,min(self.grid_size,round(filled.shape[1]/scale)))
                pooled=torch.nn.functional.adaptive_avg_pool2d(filled[None,None],(height,width))[0,0]
                counts=torch.nn.functional.adaptive_avg_pool2d(coverage[None,None],(height,width))[0,0]
                grid=torch.where(counts>0,pooled/counts.clamp_min(1e-12),0)
                valid_cells=counts.tolist()
                values=grid.tolist()
                entry['grid']={
                    'shape':[height,width],
                    'values':[[clean(values[row][col]) if valid_cells[row][col]>0 else None
                               for col in range(width)] for row in range(height)],
                    'aggregation':'mean',
                    'source_shape':source_shape,
                    'spatial_axes':list(range(max(0,value.ndim-2),value.ndim)),
                    'reduced_axes':list(range(max(0,value.ndim-2))),
                    'finite_elements':int(finite.sum().item()),
                }
                entry['message']='Bounded whole-tensor grid; original activation tensor was not retained.'
        except Exception as exc:entry['message']=f'Grid unavailable: {type(exc).__name__}'
    def operation(self,oid,func,args,kwargs):
        if len(self.operations)>=self.max_operations:return
        def compact(v):
            if v is None or type(v) in (bool,int,str):return v
            if type(v) is float:return v if math.isfinite(v) else None
            if isinstance(v,(list,tuple)) and len(v)<=8 and all(type(x) in (bool,int,float,str,type(None)) for x in v):return [compact(x) for x in v]
            raise TypeError
        entry={}
        try:
            for index,spec in enumerate(func._schema.arguments):
                value=args[index] if index<len(args) else kwargs.get(spec.name,spec.default_value)
                try:entry[spec.name]=compact(value)
                except TypeError:pass
        except AttributeError:return
        self.operations[oid]=entry
    def as_dict(self):return {'tensors':self.tensors,'operations':self.operations,'limits':{'max_tensors':self.max_tensors,'max_elements':self.max_elements,'sample_size':self.sample_size,'max_summary_elements':self.max_summary_elements,'grid_size':self.grid_size}}
