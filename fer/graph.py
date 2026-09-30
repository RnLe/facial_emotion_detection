"""Export a model's layer graph as JSON, straight from the PyTorch code: every layer with its
output shape and parameter count, and every connection, including skips and concatenations.
Used for the architecture figures (and the interactive viewer on the portfolio).

Repeated units (a residual block, a dense layer, a transformer layer) are kept as groups, so
a viewer can show a block collapsed or opened.
"""
import json
from pathlib import Path

import torch
from torch import fx, nn
from torch.fx.passes.shape_prop import ShapeProp

from .models import MODELS, NAMES, build
from .models.convnext import ConvNeXtBlock, LayerNorm2d
from .models.densenet import DenseLayer
from .models.resnet import BasicBlock
from .models.transformer import Block, DropPath
from torchvision.models.resnet import BasicBlock as TorchvisionBasicBlock

# Units drawn as one box each, openable in a viewer.
GROUPS = (BasicBlock, TorchvisionBasicBlock, DenseLayer, Block, ConvNeXtBlock)
# Layers traced as one node (not followed inside).
LEAVES = (nn.MultiheadAttention, LayerNorm2d, DropPath)


class Tracer(fx.Tracer):
    def is_leaf_module(self, m, name):
        return isinstance(m, LEAVES) or super().is_leaf_module(m, name)


def label(node, modules):
    if node.op == "call_module":
        m = modules[node.target]
        kind = type(m).__name__
        if isinstance(m, nn.Conv2d):
            groups = f", depthwise" if m.groups > 1 else ""
            return kind, f"Conv {m.kernel_size[0]}x{m.kernel_size[1]}, {m.out_channels}{', stride ' + str(m.stride[0]) if m.stride[0] > 1 else ''}{groups}"
        if isinstance(m, nn.Linear):
            return kind, f"Dense {m.out_features}"
        if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            return kind, f"{'Max' if isinstance(m, nn.MaxPool2d) else 'Avg'} pool {m.kernel_size}"
        if isinstance(m, nn.MultiheadAttention):
            return kind, f"Self-attention, {m.num_heads} heads"
        names = {"BatchNorm2d": "Batch norm", "LayerNorm": "Layer norm", "LayerNorm2d": "Layer norm", "ReLU": "ReLU",
                 "GELU": "GELU", "Dropout": "Dropout", "DropPath": "Drop path", "AdaptiveAvgPool2d": "Global avg pool",
                 "Flatten": "Flatten", "Identity": "Identity"}
        return kind, names.get(kind, kind)
    fn = getattr(node.target, "__name__", str(node.target))
    names = {"add": "Add", "iadd": "Add", "cat": "Concat", "relu": "ReLU", "flatten": "Flatten", "transpose": "Reshape",
             "softmax": "Softmax", "mul": "Multiply", "sum": "Sum", "expand": "Expand", "layer_norm": "Layer norm",
             "permute": "Reshape", "getitem": "Select"}
    return fn, names.get(fn, fn)


def group_of(target, modules):
    """The innermost GROUPS module a node belongs to, as a dotted path, or None."""
    parts = str(target).split(".") if target else []
    best = None
    for k in range(1, len(parts) + 1):
        path = ".".join(parts[:k])
        if path in modules and isinstance(modules[path], GROUPS):
            best = path
    return best


def export(name):
    model = build(name).eval()
    modules = dict(model.named_modules())
    graph = Tracer().trace(model)
    gm = fx.GraphModule(model, graph)
    ShapeProp(gm).propagate(torch.randn(1, 1, 48, 48))

    nodes, edges, ids = [], [], {}
    for node in gm.graph.nodes:
        if node.op in ("get_attr",):
            continue
        meta = node.meta.get("tensor_meta")
        shape = list(meta.shape[1:]) if meta is not None and hasattr(meta, "shape") else None
        if node.op == "placeholder":
            kind, text = "Input", "Input 48x48"
        elif node.op == "output":
            kind, text = "Output", "7 emotions"
        else:
            kind, text = label(node, modules)
            params_read = [a.target for a in node.all_input_nodes if a.op == "get_attr"]
            if any("pos" in t for t in params_read):
                kind, text = "add", "Add positions"
            elif any("cls" in t for t in params_read):
                kind, text = "Token", "Class token"
            elif node.op == "call_function" and getattr(node.target, "__name__", "") == "getitem" and isinstance(node.args[1], tuple) and node.args[1][-1] == 0:
                text = "Take class token"
        target = node.target if node.op == "call_module" else node.meta.get("nn_module_stack") and list(node.meta["nn_module_stack"].values())[-1][0]
        params = sum(p.numel() for p in modules[node.target].parameters(recurse=False)) if node.op == "call_module" else 0
        if node.op == "call_module":
            params = sum(p.numel() for p in modules[node.target].parameters())
        ids[node] = len(nodes)
        nodes.append({"id": len(nodes), "kind": kind, "label": text, "shape": shape, "params": params,
                      "group": group_of(target if isinstance(target, str) else None, modules)})
        for arg in node.all_input_nodes:
            if arg in ids and text != "Class token":  # the class token only borrows the batch size
                edges.append([ids[arg], ids[node]])

    # Drop bookkeeping nodes that carry no tensor (reading a shape), joining their neighbours.
    drop = {n["id"] for n in nodes if n["kind"] == "getattr" or (n["kind"] == "getitem" and n["shape"] is None)}
    for d in drop:
        ins = [a for a, b in edges if b == d]
        outs = [b for a, b in edges if a == d]
        edges = [e for e in edges if d not in e] + [[a, b] for a in ins for b in outs if a not in drop]
    edges = [e for e in edges if e[0] not in drop and e[1] not in drop]
    keep = [n for n in nodes if n["id"] not in drop]
    renumber = {n["id"]: k for k, n in enumerate(keep)}
    nodes = [{**n, "id": renumber[n["id"]]} for n in keep]
    edges = sorted({(renumber[a], renumber[b]) for a, b in edges})

    groups = {}
    for path, m in modules.items():
        if isinstance(m, GROUPS):
            groups[path] = {"kind": type(m).__name__, "params": sum(p.numel() for p in m.parameters())}
    return {"model": name, "name": NAMES[name], "params": sum(p.numel() for p in model.parameters()),
            "nodes": nodes, "edges": edges, "groups": groups}


if __name__ == "__main__":
    out = Path("results/graphs")
    out.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        g = export(name)
        (out / f"{name}.json").write_text(json.dumps(g))
        print(f"{name:18s} {len(g['nodes']):4d} nodes {len(g['edges']):4d} edges {len(g['groups']):3d} groups")
