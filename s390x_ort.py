from collections import defaultdict
import math
import onnx

class _InputMeta:
    def __init__(self, name: str):
        self.name = name

class InferenceSession:
    """Native s390x ONNX TreeEnsembleClassifier execution engine."""
    def __init__(self, onnx_path: str):
        model = onnx.load(onnx_path)
        self.input_name = model.graph.input[0].name
        node = next(n for n in model.graph.node if n.op_type == "TreeEnsembleClassifier")

        attrs = {}
        for a in node.attribute:
            if a.type == onnx.AttributeProto.INTS:
                attrs[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.FLOATS:
                attrs[a.name] = list(a.floats)
            elif a.type == onnx.AttributeProto.STRINGS:
                attrs[a.name] = [s.decode("utf-8") for s in a.strings]
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode("utf-8")

        class_ids = attrs.get("class_ids", [0] * len(attrs["class_weights"]))
        has_class_1 = 1 in set(class_ids)
        leaf_weights = defaultdict(float)
        for tid, nid, cid, w in zip(
            attrs["class_treeids"], attrs["class_nodeids"], class_ids, attrs["class_weights"]
        ):
            if not has_class_1 or cid == 1:
                leaf_weights[(tid, nid)] += w

        trees = defaultdict(dict)
        for tid, nid, fid, val, mode, t_id, f_id in zip(
            attrs["nodes_treeids"],
            attrs["nodes_nodeids"],
            attrs["nodes_featureids"],
            attrs["nodes_values"],
            attrs["nodes_modes"],
            attrs["nodes_truenodeids"],
            attrs["nodes_falsenodeids"],
        ):
            trees[tid][nid] = (mode, fid, val, t_id, f_id, leaf_weights[(tid, nid)])

        base_vals = attrs.get("base_values", [0.0])
        init_val = base_vals[1] if (has_class_1 and len(base_vals) > 1) else (base_vals[0] if base_vals else 0.0)

        lines = ["def _score_tx(x):", f"    s = {float(init_val)!r}"]

        def emit_node(tid, nid, indent):
            mode, fid, val, t_id, f_id, w = trees[tid][nid]
            pad = "    " * indent
            if mode == "LEAF":
                lines.append(f"{pad}s += {float(w)!r}")
            else:
                op = "<=" if "LEQ" in mode else "<" if "LT" in mode else ">"
                lines.append(f"{pad}if x[{fid}] {op} {float(val)!r}:")
                emit_node(tid, t_id, indent + 1)
                lines.append(f"{pad}else:")
                emit_node(tid, f_id, indent + 1)

        for tid in sorted(trees.keys()):
            emit_node(tid, 0, 1)

        lines.append("    s = 500.0 if s > 500.0 else (-500.0 if s < -500.0 else s)")
        lines.append("    return 1.0 / (1.0 + math.exp(-s))")
        ns = {"math": math}
        exec("\n".join(lines), ns)
        self._score_tx = ns["_score_tx"]

    def get_inputs(self):
        return [_InputMeta(self.input_name)]

    def run(self, output_names, input_feed):
        tx = input_feed[self.input_name][0].tolist()
        p1 = self._score_tx(tx)
        return [[1 if p1 >= 0.5 else 0], [{0: 1.0 - p1, 1: p1}]]