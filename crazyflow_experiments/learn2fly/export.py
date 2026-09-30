"""Export Flax ActorNet parameters to a C header."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from crazyflow_experiments.core.agents.td3_agent import load_actor_params
from crazyflow_experiments.learn2fly import MODEL_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = MODEL_DIR / "td3_rprv_model.ckpt"
OUTPUT_PATH = MODEL_DIR / "td3_rprv_policy_params.h"
HEADER_GUARD = "ACTOR_PARAMS_H"


def _dense_layers(actor_tree: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    names = [name for name in actor_tree if name.startswith("Dense_")]
    names = sorted(names, key=lambda name: int(name.split("_", 1)[1]))

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for name in names:
        node = actor_tree[name]
        kernel = np.asarray(node["kernel"], dtype=np.float32)
        bias = np.asarray(node["bias"], dtype=np.float32)

        if kernel.ndim != 2:
            raise ValueError(f"{name}.kernel expected 2D, got {kernel.shape}")
        if bias.shape != kernel.shape[1:]:
            raise ValueError(
                f"{name}.bias shape {bias.shape} not compatible with kernel {kernel.shape}"
            )
        layers.append((kernel, bias))

    if not layers:
        raise ValueError("No Dense_* modules found in actor params tree.")
    return layers


def _float_literal(value: float) -> str:
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal += ".0"

    return f"{literal}f"


def _format_array_values(values: np.ndarray, columns: int = 8) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    lines = [f"  {_float_literal(float(value))}" for value in flat]

    grouped_lines = []
    for start in range(0, len(lines), columns):
        chunk = ", ".join(lines[start : start + columns])
        if start + columns < len(lines):
            chunk += ","
        grouped_lines.append(f"      {chunk}")

    return "\n".join(grouped_lines) + "\n"


def _validate_layer_chain(layers: list[tuple[np.ndarray, np.ndarray]]):
    for idx, ((kernel, _), (next_kernel, _)) in enumerate(zip(layers, layers[1:])):
        if kernel.shape[1] != next_kernel.shape[0]:
            raise ValueError(
                f"Dense_{idx} output size {kernel.shape[1]} does not match Dense_{idx + 1} input "
                f"size {next_kernel.shape[0]}"
            )


def _format_layer(layer_index: int, kernel: np.ndarray, bias: np.ndarray) -> list[str]:
    in_dim, out_dim = kernel.shape
    weight_name = f"actor_W{layer_index}"
    bias_name = f"actor_b{layer_index}"
    weight_dim_name = f"DIM_W{layer_index}"

    return [
        f"\n  // Layer {layer_index}: {in_dim} -> {out_dim}\n",
        "  enum\n  {\n",
        f"    ACTOR_L{layer_index}_IN = {in_dim},\n",
        f"    ACTOR_L{layer_index}_OUT = {out_dim},\n",
        f"    {weight_dim_name} = ACTOR_L{layer_index}_IN * ACTOR_L{layer_index}_OUT\n",
        "  };\n",
        f"  static const __fp16 {weight_name}[{weight_dim_name}] = {{\n",
        _format_array_values(kernel),
        "  };\n",
        f"  static const __fp16 {bias_name}[ACTOR_L{layer_index}_OUT] = {{\n",
        _format_array_values(bias),
        "  };\n",
    ]


def export_header():
    """Export the actor network parameters to a C header file."""
    actor_tree = load_actor_params(MODEL_PATH)["params"]
    layers = _dense_layers(actor_tree)
    _validate_layer_chain(layers)

    input_size = layers[0][0].shape[0]
    output_size = layers[-1][0].shape[1]
    hidden_sizes = [kernel.shape[1] for kernel, _ in layers[:-1]]
    hidden_size = hidden_sizes[0] if hidden_sizes else 0
    num_hidden_layers = len(hidden_sizes)
    num_dense_layers = len(layers)
    layer_dims = [input_size, *(kernel.shape[1] for kernel, _ in layers)]
    total_params = sum(kernel.size + bias.size for kernel, bias in layers)
    total_bytes = total_params * 2

    lines = [
        f"// Auto-generated from: {MODEL_PATH.name}\n",
        "// ActorNet params for deterministic inference (mean only).\n",
        "//\n",
        "// Actor network metadata (derived from Dense params):\n",
        f"// - input_size   : {input_size}\n",
        f"// - output_size  : {output_size}\n",
        f"// - num_layers   : {num_hidden_layers} (hidden Dense layers)\n",
        f"// - hidden_sizes : {hidden_sizes}\n",
        f"// - dense_layers : {num_dense_layers} (including output Dense)\n",
        f"// - layer_dims   : {' -> '.join(str(dim) for dim in layer_dims)}\n",
        f"// - total_params : {total_params} (kernel+bias only)\n",
        f"// - param_bytes  : {total_bytes} (float16)\n",
        "//\n",
        "// Storage convention:\n",
        "// - Flax Dense kernel shape is (in_features, out_features)\n",
        "// - We store actor_Wk as row-major flattened kernel:\n",
        "//     actor_Wk[i*out + j] == kernel[i, j]\n",
        "// - Bias stored as actor_bk[j]\n",
        "// - Parameters stored as float16 (__fp16) to save bandwidth over CRTP\n",
        "//\n",
        f"\n#ifndef {HEADER_GUARD}\n#define {HEADER_GUARD}\n\n",
        '#ifdef __cplusplus\nextern "C"\n{\n#endif\n\n',
        "#ifndef OBS_DIM\n",
        f"#define OBS_DIM {input_size}\n",
        f"#elif OBS_DIM != {input_size}\n",
        '#error "OBS_DIM does not match the exported actor input size"\n',
        "#endif\n\n",
        "// Use enums for compile-time constants in C array declarations.\n",
        "enum\n  {\n",
        f"  ACTOR_INPUT_SIZE = {input_size},\n",
        f"  ACTOR_OUTPUT_SIZE = {output_size},\n",
        f"  ACTOR_HIDDEN_SIZE = {hidden_size},\n",
        f"  ACTOR_NUM_LAYERS = {num_hidden_layers},\n",
        f"  ACTOR_NUM_DENSE = {num_dense_layers},\n",
        f"  ACTOR_TOTAL_PARAMS = {total_params},\n",
        f"  ACTOR_PARAM_BYTES = {total_bytes}\n",
        "};\n",
        "\n",
        f"#define ACTOR_NUM_LAYERS {num_hidden_layers}\n",
        "\n",
    ]

    for layer_index, (kernel, bias) in enumerate(layers):
        lines.extend(_format_layer(layer_index, kernel, bias))

    lines.extend(["\n#ifdef __cplusplus\n}\n#endif\n\n", f"#endif // {HEADER_GUARD}\n"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_header()
    logger.info(f"Wrote header: {OUTPUT_PATH}")
