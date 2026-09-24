"""One observed forward pass; hooks and dispatcher capture remain separate."""
import weakref
import traceback
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from p2i.ir import TensorInfo, OperationNode, RuntimeCall, SourceLocation
from .tree import tensors

class Recorder(TorchDispatchMode):
    def __init__(self, ir, identities, inspection=None):
        super().__init__()
        self.ir, self.identities = ir, identities
        self.lookup, self.stack, self.handles = {}, [], []
        self.tensor_map = {}
        self.nodes = {n.id: n for n in ir.modules}
        self.enabled = True
        self.inspection = inspection

    def tensor(self, value, producer=None, fresh=False):
        found = self.lookup.get(id(value))
        previous = found[1] if found and found[0]() is value else None
        if previous is not None and not fresh:
            return previous
        tid = f"rt{len(self.ir.tensors)}"
        info = TensorInfo(id=tid, shape=[int(d) for d in value.shape], dtype=str(value.dtype),
                          device=str(value.device), requires_grad=value.requires_grad,
                          producer_id=producer, alias_of=previous.id if previous and fresh else None)
        self.lookup[id(value)] = (weakref.ref(value), info)
        self.ir.tensors.append(info)
        self.tensor_map[tid] = info
        if self.inspection is not None:self.inspection.tensor(info,value)
        return info

    def ids(self, tree):
        return list(dict.fromkeys(self.tensor(t).id for t in tensors(tree)))

    def pre(self, module, args, kwargs):
        mid = self.identities[id(module)]
        call = RuntimeCall(id=f"c{len(self.ir.runtime.calls)}", module_id=mid,
                           parent_call_id=self.stack[-1].id if self.stack else None,
                           sequence=len(self.ir.runtime.calls), input_tensor_ids=self.ids((args, kwargs)))
        self.ir.runtime.calls.append(call)
        self.stack.append(call)
        node = self.nodes[mid]
        node.executed = True
        node.call_ids.append(call.id)
        node.input_tensor_ids = list(dict.fromkeys(node.input_tensor_ids + call.input_tensor_ids))

    def post(self, module, args, kwargs, output):
        call = self.stack.pop()
        call.output_tensor_ids = self.ids(output)
        call.completed = True
        node = self.nodes[call.module_id]
        node.output_tensor_ids = list(dict.fromkeys(node.output_tensor_ids + call.output_tensor_ids))

    def install(self, model):
        for module in model.modules():
            self.handles.append(module.register_forward_pre_hook(self.pre, with_kwargs=True))
            self.handles.append(module.register_forward_hook(self.post, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        inputs = self.ids((args, kwargs))
        result = func(*args, **kwargs)
        oid = f"r{len(self.ir.operations)}"
        if self.inspection is not None:self.inspection.operation(oid,func,args,kwargs)
        outputs = []
        seen = set()
        for tensor in tensors(result):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            info = self.tensor(tensor, producer=oid, fresh=True)
            if info.id not in outputs:
                outputs.append(info.id)
        source = None
        for frame in reversed(traceback.extract_stack(limit=40)[:-1]):
            if '/p2i/analyze/' not in frame.filename and '/torch/' not in frame.filename:
                source = SourceLocation(file=frame.filename, line=frame.lineno, function=frame.name)
                break
        call = self.stack[-1] if self.stack else None
        op = OperationNode(id=oid, graph="runtime", op_type=str(func), display_name=str(func),
                           parent_module_id=call.module_id if call else None,
                           input_tensor_ids=inputs, output_tensor_ids=outputs, source=source,
                           executed=True, call_id=call.id if call else None, sequence=len(self.ir.operations))
        self.ir.operations.append(op)
        for tid in inputs:
            self.tensor_map[tid].consumer_ids.append(oid)
        return result
