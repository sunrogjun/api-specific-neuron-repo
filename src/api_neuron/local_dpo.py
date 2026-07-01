from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from .common import get_layers


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class NeuronLocalEdit(nn.Module):
    def __init__(self, base_mlp: nn.Module, neuron_ids: List[int], mode: str):
        super().__init__()
        self.base = base_mlp
        self.mode = mode
        self.enabled = True
        self.register_buffer("neuron_idx", torch.tensor(sorted(set(neuron_ids)), dtype=torch.long))

        if hasattr(base_mlp, "c_proj") and hasattr(base_mlp, "c_fc"):
            self.kind = "starcoder"
            hidden_size = base_mlp.c_fc.in_features
            self.act_fn = base_mlp.act
        elif hasattr(base_mlp, "down_proj") and hasattr(base_mlp, "up_proj") and hasattr(base_mlp, "gate_proj"):
            self.kind = "llama"
            hidden_size = base_mlp.up_proj.in_features
            self.act_fn = getattr(base_mlp, "act_fn", None) or getattr(base_mlp, "act", None)
        else:
            raise ValueError(f"Unsupported MLP type: {base_mlp.__class__.__name__}")

        count = len(neuron_ids)
        dtype = next(base_mlp.parameters()).dtype
        self.delta_out = nn.Parameter(torch.zeros(count, hidden_size, dtype=dtype))
        self.delta_in = nn.Parameter(torch.zeros(count, hidden_size, dtype=dtype))
        self.delta_bias = nn.Parameter(torch.zeros(count, dtype=dtype))

    def forward(self, hidden_states):
        if not self.enabled:
            return self.base(hidden_states)

        if self.kind == "starcoder":
            pre = self.base.c_fc(hidden_states)
            if self.mode == "full":
                update = hidden_states @ self.delta_in.T + self.delta_bias
                pre = pre.clone()
                pre.index_add_(dim=-1, index=self.neuron_idx, source=update)
            act = self.act_fn(pre)
            out = self.base.c_proj(act)
            out = out + act.index_select(dim=-1, index=self.neuron_idx) @ self.delta_out
            dropout = getattr(self.base, "dropout", None)
            return dropout(out) if dropout is not None else out

        gate = self.base.gate_proj(hidden_states)
        up = self.base.up_proj(hidden_states)
        act = self.act_fn(gate) * up
        if self.mode == "full":
            update = hidden_states @ self.delta_in.T + self.delta_bias
            act = act.clone()
            act.index_add_(dim=-1, index=self.neuron_idx, source=update)
        out = self.base.down_proj(act)
        return out + act.index_select(dim=-1, index=self.neuron_idx) @ self.delta_out


def attach_local_edits(model, neurons_by_layer: Dict[int, List[int]], mode: str) -> None:
    for layer_idx, neuron_ids in neurons_by_layer.items():
        layer = get_layers(model)[layer_idx]
        if isinstance(layer.mlp, NeuronLocalEdit):
            continue
        layer.mlp = NeuronLocalEdit(layer.mlp, neuron_ids, mode)


def iter_local_edits(model) -> Iterable[tuple[int, NeuronLocalEdit]]:
    for layer_idx, layer in enumerate(get_layers(model)):
        if isinstance(layer.mlp, NeuronLocalEdit):
            yield layer_idx, layer.mlp


def set_local_edits_enabled(model, enabled: bool) -> None:
    for _, edit in iter_local_edits(model):
        edit.enabled = enabled


def trainable_parameters(model) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for _, edit in iter_local_edits(model):
        edit.delta_out.requires_grad_(True)
        params.append(edit.delta_out)
        if edit.mode == "full":
            edit.delta_in.requires_grad_(True)
            edit.delta_bias.requires_grad_(True)
            params.extend([edit.delta_in, edit.delta_bias])
        else:
            edit.delta_in.requires_grad_(False)
            edit.delta_bias.requires_grad_(False)
    return params


def export_edit_payload(model, model_path: str) -> dict:
    layers = []
    mode = None
    for layer_idx, edit in iter_local_edits(model):
        mode = edit.mode
        layers.append(
            {
                "layer": layer_idx,
                "neuron_ids": edit.neuron_idx.detach().cpu(),
                "delta_out": edit.delta_out.detach().cpu(),
                "delta_in": edit.delta_in.detach().cpu(),
                "delta_bias": edit.delta_bias.detach().cpu(),
            }
        )
    return {"model_path": model_path, "mode": mode, "layers": layers}


def normalize_edit_mode(mode: str | None) -> str:
    text = (mode or "").strip().lower().replace("-", "_")
    if text in {"down_only", "wdownonly", "ffn2_weight", "ffn2"}:
        return "down_only"
    if text == "full":
        return "full"
    return "full"


def infer_neurons_by_layer_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[int, List[int]]:
    by_layer: Dict[int, List[int]] = {}
    for key, value in state_dict.items():
        if not str(key).endswith("neuron_idx"):
            continue
        match = LAYER_RE.search(str(key))
        if not match:
            continue
        layer_idx = int(match.group(1))
        try:
            indices = value.detach().cpu().tolist()
        except Exception:
            continue
        if isinstance(indices, int):
            indices = [indices]
        by_layer[layer_idx] = [int(index) for index in indices]
    return by_layer


def export_state_dict_payload(
    model,
    model_path: str,
    neuron_file: Path | str,
    top_k: int,
    mode: str,
) -> dict:
    keep = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "delta_" in key or key.endswith("neuron_idx")
    }
    return {
        "meta": {
            "model_path": model_path,
            "neuron_file": str(neuron_file),
            "neuron_json": str(neuron_file),
            "top_k": int(top_k),
            "delta_mode": normalize_edit_mode(mode),
        },
        "state_dict": keep,
    }


def export_ffn2_weight_payload(
    model,
    model_path: str,
    neuron_file: Path | str,
    top_k: int,
) -> dict:
    layers = []
    for layer_idx, edit in iter_local_edits(model):
        cols = [int(x) for x in edit.neuron_idx.detach().cpu().tolist()]
        if not cols:
            continue
        # delta_out has shape [count, hidden]; column deltas use [hidden, count].
        layers.append(
            {
                "layer": int(layer_idx),
                "cols": cols,
                "delta": edit.delta_out.detach().cpu().T.contiguous(),
            }
        )
    return {
        "meta": {
            "model_path": model_path,
            "neuron_file": str(neuron_file),
            "neuron_json": str(neuron_file),
            "top_k": int(top_k),
            "delta_mode": "down_only",
        },
        "ffn2_weight": {"layers": layers},
    }


def apply_state_dict_payload(
    model,
    payload: dict,
    mode_hint: str | None = None,
) -> Tuple[str, Sequence[str], Sequence[str]]:
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid state_dict payload: expected dict, got {type(state_dict)}")

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    mode = normalize_edit_mode(meta.get("delta_mode") if isinstance(meta, dict) else None)
    if mode == "full" and mode_hint:
        mode = normalize_edit_mode(mode_hint)

    neurons_by_layer = infer_neurons_by_layer_from_state_dict(state_dict)
    if not neurons_by_layer:
        raise ValueError("Could not infer neuron_idx buffers from state_dict payload.")

    attach_local_edits(model, neurons_by_layer, mode=mode)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return mode, missing, unexpected


def apply_edit_payload(model, payload: dict) -> None:
    neurons_by_layer = {
        int(layer["layer"]): [int(x) for x in layer["neuron_ids"].tolist()]
        for layer in payload["layers"]
    }
    attach_local_edits(model, neurons_by_layer, payload["mode"])
    edits = {layer_idx: edit for layer_idx, edit in iter_local_edits(model)}
    for layer in payload["layers"]:
        edit = edits[int(layer["layer"])]
        edit.delta_out.data.copy_(layer["delta_out"].to(edit.delta_out.dtype))
        edit.delta_in.data.copy_(layer["delta_in"].to(edit.delta_in.dtype))
        edit.delta_bias.data.copy_(layer["delta_bias"].to(edit.delta_bias.dtype))
