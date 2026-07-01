from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    dtype: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    torch_device = resolve_device(device)
    torch_dtype = resolve_dtype(torch_device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    model = model.to(torch_device)
    model.eval()
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, torch_device


def get_layers(model) -> Sequence:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported transformer layout.")


def get_ffn2_module(layer) -> Tuple[torch.nn.Module, str]:
    mlp = layer.mlp
    if hasattr(mlp, "c_proj") and hasattr(mlp, "c_fc"):
        return mlp.c_proj, "starcoder"
    if hasattr(mlp, "down_proj") and hasattr(mlp, "up_proj"):
        return mlp.down_proj, "llama"
    raise ValueError(f"Unsupported MLP type: {mlp.__class__.__name__}")


def get_ffn2_weight(layer) -> torch.Tensor:
    module, _ = get_ffn2_module(layer)
    return module.weight.detach().T


def get_unembedding(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach()
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise ValueError("Could not find output embedding matrix.")
    return output.weight.detach()


def build_seed_direction(tokenizer, model, seeds: Sequence[str]) -> torch.Tensor:
    unembedding = get_unembedding(model).float()
    vectors = []
    for seed in seeds:
        token_ids = tokenizer(seed, add_special_tokens=False).input_ids
        if token_ids:
            vectors.append(unembedding[token_ids].mean(dim=0))
    if not vectors:
        raise ValueError("No valid seed tokens found.")
    return torch.stack(vectors).mean(dim=0)


def clean_token(token: str) -> str:
    while token.startswith(("Ġ", "▁", " ")):
        token = token[1:]
    return token.strip()


def is_readable_token(token: str) -> bool:
    if not token:
        return False
    if any(ch not in string.printable for ch in token):
        return False
    return True


def decode_direction(tokenizer, model, direction: torch.Tensor, top_n: int = 8) -> List[str]:
    scores = torch.matmul(get_unembedding(model).float(), direction.float())
    values, indices = torch.topk(scores, k=min(scores.numel(), top_n * 6))
    decoded: List[str] = []
    for token_id in indices.tolist():
        token = clean_token(tokenizer.convert_ids_to_tokens([token_id])[0])
        if is_readable_token(token):
            decoded.append(token)
        if len(decoded) >= top_n:
            break
    return decoded


def load_top_neurons(path: Path, top_k: int = 200) -> List[Tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs: List[Tuple[int, int]] = []
    seen = set()
    for item in data["global_top_neurons"][:top_k]:
        pair = (int(item["layer"]), int(item["neuron"]))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def group_neurons(neurons: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = {}
    for layer, neuron in neurons:
        grouped.setdefault(int(layer), []).append(int(neuron))
    return {layer: sorted(set(indices)) for layer, indices in grouped.items()}


def first_identifier(text: str) -> str:
    token = []
    for char in (text or "").lstrip():
        if char.isalnum() or char == "_":
            token.append(char)
        elif token:
            break
    return "".join(token)


def first_api_chain(text: str) -> str:
    text = (text or "").lstrip().lstrip(".")
    chain = []
    for char in text:
        if char.isalnum() or char in "._":
            chain.append(char)
        elif chain:
            break
    result = "".join(chain)
    while result.startswith("torch."):
        result = result[len("torch.") :]
    return result


def api_matches(predicted: str, target: str) -> bool:
    if not predicted or not target:
        return False
    if predicted == target:
        return True
    if "." not in target and predicted.endswith("." + target):
        return True
    return False
