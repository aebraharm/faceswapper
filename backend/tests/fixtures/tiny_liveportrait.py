"""Builds a tiny, synthetic ONNX graph bundle that satisfies the LivePortrait ABI.

This is NOT the real LivePortrait model -- it is a few KB of trivial ops used only
so CI can exercise `LivePortraitOnnxTransformer`'s session plumbing, keypoint math
and caching behavior without downloading ~537 MB of real weights on every run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def _save(graph_name: str, inputs, outputs, nodes, initializers, path: Path) -> None:
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _const(name: str, values, dtype=TensorProto.INT64):
    array = np.asarray(values, dtype=np.int64 if dtype == TensorProto.INT64 else np.float32)
    return helper.make_tensor(name, dtype, array.shape, array.flatten().tolist())


def _scalar_from_mean(input_name: str, offset: float, out_name: str, prefix: str):
    """mean(input) + offset, as a rank-0-ish [1,1] tensor, expressed as ONNX nodes."""
    mean_name = f"{prefix}_mean"
    offset_name = f"{prefix}_offset"
    reshaped_name = f"{prefix}_reshape"
    shape_name = f"{prefix}_shape11"
    nodes = [
        helper.make_node("ReduceMean", [input_name], [mean_name], keepdims=0),
        helper.make_node("Add", [mean_name, offset_name], [reshaped_name + "_sum"]),
        helper.make_node("Reshape", [reshaped_name + "_sum", shape_name], [out_name]),
    ]
    initializers = [
        helper.make_tensor(offset_name, TensorProto.FLOAT, [], [offset]),
        _const(shape_name, [1, 1]),
    ]
    return nodes, initializers


def _expand_scalar(scalar_name: str, target_shape: list[int], out_name: str, prefix: str):
    shape_name = f"{prefix}_targetshape"
    reshape_shape_name = f"{prefix}_onesshape"
    ones_rank_name = f"{prefix}_onesrank"
    nodes = [
        helper.make_node("Reshape", [scalar_name, reshape_shape_name], [ones_rank_name]),
        helper.make_node("Expand", [ones_rank_name, shape_name], [out_name]),
    ]
    initializers = [
        _const(reshape_shape_name, [1] * len(target_shape)),
        _const(shape_name, target_shape),
    ]
    return nodes, initializers


def build_appearance_feature_extractor(path: Path, feature_shape=(1, 32, 16, 64, 64)) -> None:
    img = helper.make_tensor_value_info("img", TensorProto.FLOAT, [1, 3, 256, 256])
    feature = helper.make_tensor_value_info("feature_3d", TensorProto.FLOAT, list(feature_shape))
    nodes, inits = _scalar_from_mean("img", 0.1, "scalar", "f")
    expand_nodes, expand_inits = _expand_scalar("scalar", list(feature_shape), "feature_3d", "f")
    _save(
        "appearance_feature_extractor",
        [img],
        [feature],
        nodes + expand_nodes,
        inits + expand_inits,
        path,
    )


def build_motion_extractor(path: Path, num_kp: int = 21) -> None:
    img = helper.make_tensor_value_info("img", TensorProto.FLOAT, [1, 3, 256, 256])
    outputs = [
        helper.make_tensor_value_info("pitch", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("yaw", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("roll", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("t", TensorProto.FLOAT, [1, 3]),
        helper.make_tensor_value_info("exp", TensorProto.FLOAT, [1, num_kp * 3]),
        helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("kp", TensorProto.FLOAT, [1, num_kp * 3]),
    ]
    all_nodes, all_inits = [], []
    # Small distinct offsets so pitch/yaw/roll/scale are not numerically identical;
    # values stay small so sin/cos/rotation math stays well-conditioned.
    scalar_specs = [("pitch", 2.0), ("yaw", -3.0), ("roll", 1.0), ("scale", 1.3)]
    scalars = {}
    for out_name, offset in scalar_specs:
        nodes, inits = _scalar_from_mean("img", offset, out_name, out_name)
        all_nodes += nodes
        all_inits += inits
        scalars[out_name] = out_name

    t_nodes, t_inits = _expand_scalar(scalars["pitch"], [1, 3], "t", "t")
    exp_nodes, exp_inits = _expand_scalar(scalars["yaw"], [1, num_kp * 3], "exp", "exp")
    kp_nodes, kp_inits = _expand_scalar(scalars["roll"], [1, num_kp * 3], "kp", "kp")
    all_nodes += t_nodes + exp_nodes + kp_nodes
    all_inits += t_inits + exp_inits + kp_inits

    _save("motion_extractor", [img], outputs, all_nodes, all_inits, path)


def build_warping_spade(
    path: Path,
    feature_shape=(1, 32, 16, 64, 64),
    num_kp: int = 21,
    output_size: int = 256,
) -> None:
    feature = helper.make_tensor_value_info("feature_3d", TensorProto.FLOAT, list(feature_shape))
    kp_source = helper.make_tensor_value_info("kp_source", TensorProto.FLOAT, [1, num_kp, 3])
    kp_driving = helper.make_tensor_value_info("kp_driving", TensorProto.FLOAT, [1, num_kp, 3])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, output_size, output_size])

    nodes = [
        helper.make_node("ReduceMean", ["feature_3d"], ["f_mean"], keepdims=0),
        helper.make_node("ReduceMean", ["kp_source"], ["s_mean"], keepdims=0),
        helper.make_node("ReduceMean", ["kp_driving"], ["d_mean"], keepdims=0),
        helper.make_node("Add", ["f_mean", "s_mean"], ["fs"]),
        helper.make_node("Add", ["fs", "d_mean"], ["fsd"]),
        helper.make_node("Sigmoid", ["fsd"], ["sig"]),
    ]
    expand_nodes, expand_inits = _expand_scalar("sig", [1, 3, output_size, output_size], "out", "w")
    _save("warping_spade", [feature, kp_source, kp_driving], [out], nodes + expand_nodes, expand_inits, path)


def build_tiny_liveportrait_bundle(directory: Path, warping_output_size: int = 256) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "appearance_feature_extractor.onnx": directory / "appearance_feature_extractor.onnx",
        "motion_extractor.onnx": directory / "motion_extractor.onnx",
        "warping_spade.onnx": directory / "warping_spade.onnx",
    }
    build_appearance_feature_extractor(paths["appearance_feature_extractor.onnx"])
    build_motion_extractor(paths["motion_extractor.onnx"])
    build_warping_spade(paths["warping_spade.onnx"], output_size=warping_output_size)
    return paths
