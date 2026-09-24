"""Static graphs have their own tensors and never claim runtime execution."""
import re
import torch
from torch.fx import Node
from p2i.ir import OperationNode, TensorInfo, SourceLocation
from .tree import tensors

def ingest(graph_module, ir, paths, graph):
    refs = {}
    prefix = "e" if graph == "export" else "f"
    tensor_map = {t.id: t for t in ir.tensors}
    for index, node in enumerate(graph_module.graph.nodes):
        inputs = []
        def collect(obj):
            if isinstance(obj, Node):
                inputs.extend(refs.get(obj, []))
            return obj
        torch.fx.node.map_arg((node.args, node.kwargs), collect)
        inputs = list(dict.fromkeys(inputs))
        values = tensors(node.meta.get("val"))
        if not values and "tensor_meta" in node.meta:
            def metadata_leaves(obj):
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    return [obj]
                if isinstance(obj, dict):
                    return [v for item in obj.values() for v in metadata_leaves(item)]
                if isinstance(obj, (tuple, list)):
                    return [v for item in obj for v in metadata_leaves(item)]
                return []
            values = metadata_leaves(node.meta["tensor_meta"])
        outputs = []
        oid = f"{prefix}{index}"
        is_op = node.op in ("call_function", "call_method", "call_module")
        for j, value in enumerate(values):
            tid = f"{prefix}t{index}_{j}"
            shape = [d if type(d) is int else str(d) for d in value.shape]
            t = TensorInfo(id=tid, graph=graph, shape=shape, dtype=str(value.dtype),
                           device=str(value.device) if hasattr(value, "device") else None, requires_grad=value.requires_grad,
                           producer_id=oid if is_op else None)
            ir.tensors.append(t)
            tensor_map[tid] = t
            outputs.append(tid)
        refs[node] = outputs
        if not is_op:
            continue
        parent = None
        stack = node.meta.get("nn_module_stack", {})
        if stack:
            path = list(stack.values())[-1][0]
            parent = paths.get(path)
        if node.op == "call_module":
            parent = paths.get(str(node.target))
        source = None
        matches = re.findall(r'File "([^"\n]+)", line (\d+), in ([^\n]+)', node.meta.get("stack_trace", ""))
        if matches:
            file, line, function = matches[-1]
            source = SourceLocation(file=file, line=int(line), function=function)
        target = str(node.target)
        op = OperationNode(id=oid, graph=graph, op_type=target, display_name=target,
                           parent_module_id=parent, input_tensor_ids=inputs,
                           output_tensor_ids=outputs, source=source, sequence=index)
        ir.operations.append(op)
        for tid in inputs:
            tensor_map[tid].consumer_ids.append(oid)
