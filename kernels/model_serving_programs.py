"""Scaled-down TVM tensor programs used by the matched compiler-search study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class ProgramCard:
    name: str
    family: str
    shape: str
    description: str
    builder: Callable[[Any], Any]
    input_factory: Callable[[Any, Any, int], List[Any]]


def program_cards() -> Dict[str, ProgramCard]:
    return {
        "llama_attention_qk_av": ProgramCard(
            name="llama_attention_qk_av",
            family="scaled-down Reasoning-Compiler experiment workload: Llama-3-8B Attention Layer",
            shape="B=1,H=8,S=128,D=64",
            description="QK reduction followed by AV reduction; softmax omitted for course-scale TIR tuning.",
            builder=compose_attention_program,
            input_factory=attention_buffers,
        ),
        "llama_mlp_swiglu": ProgramCard(
            name="llama_mlp_swiglu",
            family="scaled-down Reasoning-Compiler experiment workload: Llama-4-Scout MLP Layer",
            shape="T=128,H=512,I=2048",
            description="Gate projection, up projection, elementwise gate*up, and down projection.",
            builder=compose_swiglu_program,
            input_factory=swiglu_buffers,
        ),
        "flux_conv_large": ProgramCard(
            name="flux_conv_large",
            family="scaled-down Reasoning-Compiler experiment workload: FLUX Convolution Layer",
            shape="N=1,IC=128,H=64,W=64,OC=128,K=3,S=1,P=1",
            description="Larger convolutional operator family than the simple sanity-check Conv2D.",
            builder=compose_flux_conv_program,
            input_factory=flux_conv_buffers,
        ),
        "simplified_moe_ffn": ProgramCard(
            name="simplified_moe_ffn",
            family="scaled-down Reasoning-Compiler experiment workload: DeepSeek-R1 MoE Layer",
            shape="T=64,E=4,H=512,I=1024",
            description="Multiple expert GEMM-like branches with dense routing for a CPU-friendly MoE proxy.",
            builder=compose_moe_program,
            input_factory=moe_buffers,
        ),
    }


def compose_attention_program(tvm: Any) -> Any:
    from tvm import te

    batch, heads, seq_len, head_dim = 1, 8, 128, 64
    query = te.placeholder((batch, heads, seq_len, head_dim), name="Q", dtype="float32")
    key = te.placeholder((batch, heads, seq_len, head_dim), name="K", dtype="float32")
    value = te.placeholder((batch, heads, seq_len, head_dim), name="V", dtype="float32")
    depth_axis = te.reduce_axis((0, head_dim), name="depth_axis")
    scores = te.compute(
        (batch, heads, seq_len, seq_len),
        lambda b, h, i, j: te.sum(query[b, h, i, depth_axis] * key[b, h, j, depth_axis], axis=depth_axis),
        name="scores",
    )
    sequence_axis = te.reduce_axis((0, seq_len), name="sequence_axis")
    output = te.compute(
        (batch, heads, seq_len, head_dim),
        lambda b, h, i, d: te.sum(scores[b, h, i, sequence_axis] * value[b, h, sequence_axis, d], axis=sequence_axis),
        name="O",
    )
    prim_func = te.create_prim_func([query, key, value, output]).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": prim_func})


def compose_swiglu_program(tvm: Any) -> Any:
    from tvm import te

    tokens, hidden, intermediate = 128, 512, 2048
    token_states = te.placeholder((tokens, hidden), name="X", dtype="float32")
    gate_weights = te.placeholder((hidden, intermediate), name="W_gate", dtype="float32")
    up_weights = te.placeholder((hidden, intermediate), name="W_up", dtype="float32")
    down_weights = te.placeholder((intermediate, hidden), name="W_down", dtype="float32")
    hidden_axis_gate = te.reduce_axis((0, hidden), name="hidden_axis_gate")
    gate = te.compute(
        (tokens, intermediate),
        lambda t, i: te.sum(token_states[t, hidden_axis_gate] * gate_weights[hidden_axis_gate, i], axis=hidden_axis_gate),
        name="gate",
    )
    hidden_axis_up = te.reduce_axis((0, hidden), name="hidden_axis_up")
    up = te.compute(
        (tokens, intermediate),
        lambda t, i: te.sum(token_states[t, hidden_axis_up] * up_weights[hidden_axis_up, i], axis=hidden_axis_up),
        name="up",
    )
    gated = te.compute((tokens, intermediate), lambda t, i: gate[t, i] * up[t, i], name="gated")
    intermediate_axis = te.reduce_axis((0, intermediate), name="intermediate_axis")
    output = te.compute(
        (tokens, hidden),
        lambda t, h: te.sum(gated[t, intermediate_axis] * down_weights[intermediate_axis, h], axis=intermediate_axis),
        name="O",
    )
    prim_func = te.create_prim_func([token_states, gate_weights, up_weights, down_weights, output]).with_attr(
        "global_symbol", "main"
    )
    return tvm.IRModule({"main": prim_func})


def compose_flux_conv_program(tvm: Any) -> Any:
    from tvm import te, topi

    batch, in_channels, height, width, out_channels, kernel = 1, 128, 64, 64, 128, 3
    image = te.placeholder((batch, in_channels, height, width), name="data", dtype="float32")
    filters = te.placeholder((out_channels, in_channels, kernel, kernel), name="weight", dtype="float32")
    conv = topi.nn.conv2d_nchw(image, filters, (1, 1), (1, 1), (1, 1), "float32")
    output = te.compute(conv.shape, lambda n, c, h, w: conv[n, c, h, w] * 1.0001, name="O")
    prim_func = te.create_prim_func([image, filters, output]).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": prim_func})


def compose_moe_program(tvm: Any) -> Any:
    from tvm import te

    tokens, experts, hidden, intermediate = 64, 4, 512, 1024
    token_states = te.placeholder((tokens, hidden), name="X", dtype="float32")
    routing = te.placeholder((tokens, experts), name="Gate", dtype="float32")
    first_weights = te.placeholder((experts, hidden, intermediate), name="W1", dtype="float32")
    second_weights = te.placeholder((experts, intermediate, hidden), name="W2", dtype="float32")
    hidden_axis = te.reduce_axis((0, hidden), name="hidden_axis")
    expert_hidden = te.compute(
        (tokens, experts, intermediate),
        lambda t, e, i: te.sum(token_states[t, hidden_axis] * first_weights[e, hidden_axis, i], axis=hidden_axis),
        name="expert_hidden",
    )
    intermediate_axis = te.reduce_axis((0, intermediate), name="intermediate_axis")
    expert_output = te.compute(
        (tokens, experts, hidden),
        lambda t, e, h: te.sum(expert_hidden[t, e, intermediate_axis] * second_weights[e, intermediate_axis, h], axis=intermediate_axis),
        name="expert_output",
    )
    expert_axis = te.reduce_axis((0, experts), name="expert_axis")
    output = te.compute(
        (tokens, hidden),
        lambda t, h: te.sum(routing[t, expert_axis] * expert_output[t, expert_axis, h], axis=expert_axis),
        name="O",
    )
    prim_func = te.create_prim_func([token_states, routing, first_weights, second_weights, output]).with_attr(
        "global_symbol", "main"
    )
    return tvm.IRModule({"main": prim_func})


def _random_array(tvm: Any, np: Any, device: Any, rng: Any, shape: tuple[int, ...]) -> Any:
    return tvm.nd.array(rng.uniform(-1, 1, size=shape).astype("float32"), device)


def _zero_array(tvm: Any, np: Any, device: Any, shape: tuple[int, ...]) -> Any:
    return tvm.nd.array(np.zeros(shape, dtype="float32"), device)


def attention_buffers(tvm: Any, device: Any, seed: int) -> List[Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    return [
        _random_array(tvm, np, device, rng, (1, 8, 128, 64)),
        _random_array(tvm, np, device, rng, (1, 8, 128, 64)),
        _random_array(tvm, np, device, rng, (1, 8, 128, 64)),
        _zero_array(tvm, np, device, (1, 8, 128, 64)),
    ]


def swiglu_buffers(tvm: Any, device: Any, seed: int) -> List[Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    return [
        _random_array(tvm, np, device, rng, (128, 512)),
        _random_array(tvm, np, device, rng, (512, 2048)),
        _random_array(tvm, np, device, rng, (512, 2048)),
        _random_array(tvm, np, device, rng, (2048, 512)),
        _zero_array(tvm, np, device, (128, 512)),
    ]


def flux_conv_buffers(tvm: Any, device: Any, seed: int) -> List[Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    return [
        _random_array(tvm, np, device, rng, (1, 128, 64, 64)),
        _random_array(tvm, np, device, rng, (128, 128, 3, 3)),
        _zero_array(tvm, np, device, (1, 128, 64, 64)),
    ]


def moe_buffers(tvm: Any, device: Any, seed: int) -> List[Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    return [
        _random_array(tvm, np, device, rng, (64, 512)),
        _random_array(tvm, np, device, rng, (64, 4)),
        _random_array(tvm, np, device, rng, (4, 512, 1024)),
        _random_array(tvm, np, device, rng, (4, 1024, 512)),
        _zero_array(tvm, np, device, (64, 512)),
    ]
