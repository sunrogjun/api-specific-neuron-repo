#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(__file__).resolve().parents[5]


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "starcoder2-3b": {
        "family": "starcoder2",
        "model_path": "/data/lkl/models/StarCoder/starcoder2-3b",
        "proj_attr": "c_proj",
        "neuron_json": REPO_ROOT / "results" / "localization" / "matrix" / "starcoder2-3b.top200.json",
        "dataset": REPO_ROOT / "data" / "other_pytorch_api_completion_base_correct" / "model_base_correct_strict" / "starcoder2-3b.jsonl",
    },
    "starcoder2-7b": {
        "family": "starcoder2",
        "model_path": "/data/lkl/models/StarCoder/starcoder2-7b",
        "proj_attr": "c_proj",
        "neuron_json": REPO_ROOT / "results" / "localization" / "matrix" / "starcoder2-7b.top200.json",
        "dataset": REPO_ROOT / "data" / "other_pytorch_api_completion_base_correct" / "model_base_correct_strict" / "starcoder2-7b.jsonl",
    },
    "starcoder2-15b": {
        "family": "starcoder2",
        "model_path": "/data/lkl/models/StarCoder/starcoder2-15b",
        "proj_attr": "c_proj",
        "neuron_json": REPO_ROOT / "results" / "localization" / "matrix" / "starcoder2-15b.top200.json",
        "dataset": REPO_ROOT / "data" / "other_pytorch_api_completion_base_correct" / "model_base_correct_strict" / "starcoder2-15b.jsonl",
    },
    "deepseek-coder-6.7b-instruct": {
        "family": "deepseek-coder",
        "model_path": "/data/lkl/models/deepseek-ai/deepseek-coder-6.7b-instruct",
        "proj_attr": "down_proj",
        "neuron_json": REPO_ROOT / "results" / "localization" / "matrix" / "deepseek-coder-6.7b-instruct.top200.json",
        "dataset": REPO_ROOT / "data" / "other_pytorch_api_completion_base_correct" / "model_base_correct_strict" / "deepseek-coder-6.7b-instruct.jsonl",
    },
    "codellama-7b": {
        "family": "codellama",
        "model_path": "/data/zzl/model/CodeLlama-7b-hf",
        "proj_attr": "down_proj",
        "neuron_json": REPO_ROOT / "results" / "localization" / "matrix" / "codellama-7b.top200.json",
        "dataset": REPO_ROOT / "data" / "other_pytorch_api_completion_base_correct" / "model_base_correct_strict" / "codellama-7b.jsonl",
    },
}


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model structure: cannot locate transformer layers.")


def load_top_neurons(path: Path, top_k: int) -> list[tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("global_top_neurons") or []
    neurons: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in entries:
        pair = (int(item["layer"]), int(item["neuron"]))
        if pair in seen:
            continue
        neurons.append(pair)
        seen.add(pair)
        if len(neurons) >= top_k:
            break
    if len(neurons) != top_k:
        raise ValueError(f"Expected {top_k} neurons from {path}, got {len(neurons)}")
    return neurons


def load_dataset(path: Path, limit: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            obj = json.loads(line)
            target = str(obj.get("target_chain") or "").strip()
            prompt = str(obj.get("prompt") or "")
            if not prompt or not target:
                continue
            items.append(
                {
                    "prompt": prompt,
                    "target_chain": target,
                    "api": obj.get("api"),
                    "category": obj.get("category"),
                    "id": obj.get("id"),
                }
            )
    return items


def clean_token(tok: str) -> str:
    while tok.startswith(("Ġ", "▁", " ")):
        tok = tok[1:]
    return tok


def extract_first_identifier(text: str) -> str:
    s = (text or "").lstrip()
    if not s:
        return ""
    token = ""
    for ch in s:
        if ch.isalnum() or ch == "_":
            token += ch
            continue
        break
    return token


def register_zero_hook(model, neurons: list[tuple[int, int]], proj_attr: str):
    by_layer: dict[int, list[int]] = {}
    for layer, dim in neurons:
        by_layer.setdefault(layer, []).append(dim)
    hooks = []

    def make_hook(layer_idx: int):
        dims = by_layer[layer_idx]

        def _hook(module, inputs):
            x = inputs[0].clone()
            x[..., dims] = 0
            return (x,)

        return _hook

    for idx, layer in enumerate(get_model_layers(model)):
        if idx not in by_layer:
            continue
        proj = getattr(layer.mlp, proj_attr)
        hooks.append(proj.register_forward_pre_hook(make_hook(idx)))
    return hooks


def add_gaussian_noise_to_columns(
    model,
    neurons: list[tuple[int, int]],
    proj_attr: str,
    std: float,
    seed: int,
):
    by_layer: dict[int, list[int]] = {}
    for layer, dim in neurons:
        by_layer.setdefault(layer, []).append(dim)
    changes = []
    generator = torch.Generator(device=model.device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        layers = get_model_layers(model)
        for layer_idx, dims in by_layer.items():
            weight = getattr(layers[layer_idx].mlp, proj_attr).weight
            for dim in dims:
                orig_col = weight[:, dim].clone()
                noise = torch.normal(
                    mean=0.0,
                    std=std,
                    size=orig_col.shape,
                    device=weight.device,
                    dtype=weight.dtype,
                    generator=generator,
                )
                weight[:, dim] = orig_col + noise
                changes.append((weight, dim, orig_col))
    return changes


def restore_columns(changes):
    with torch.no_grad():
        for weight, dim, orig_col in changes:
            weight[:, dim] = orig_col


def sample_layer_matched_random_runs(
    *,
    top_neurons: list[tuple[int, int]],
    num_layers: int,
    intermediate_dim: int,
    num_runs: int,
    seed: int,
) -> list[dict[str, Any]]:
    layer_counts = Counter(layer for layer, _ in top_neurons)
    exclude_by_layer: dict[int, set[int]] = {}
    for layer, dim in top_neurons:
        exclude_by_layer.setdefault(layer, set()).add(dim)

    runs = []
    for run_idx in range(1, num_runs + 1):
        rng = random.Random(seed + run_idx - 1)
        neurons: list[tuple[int, int]] = []
        for layer in sorted(layer_counts):
            count = int(layer_counts[layer])
            excluded = exclude_by_layer.get(layer, set())
            population = [dim for dim in range(intermediate_dim) if dim not in excluded]
            if count > len(population):
                raise ValueError(
                    f"Layer {layer} has only {len(population)} available dims after exclusion, need {count}."
                )
            for dim in rng.sample(population, count):
                neurons.append((layer, dim))
        runs.append(
            {
                "run_index": run_idx,
                "seed": seed + run_idx - 1,
                "neurons": neurons,
            }
        )
    return runs


def load_model_and_tokenizer(model_path: str, device: torch.device, torch_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer at {model_path} has neither pad nor eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def cleanup_model(model=None, tokenizer=None):
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> tuple[torch.Tensor, list[str], list[str]]:
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx is None or max_ctx > 1_000_000:
        max_ctx = getattr(model.config, "n_positions", 2048)

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_ctx),
        return_token_type_ids=False,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = (input_ids != int(tokenizer.pad_token_id)).long()
    else:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    if not out.scores:
        raise RuntimeError("Generation did not return first-step scores.")
    log_probs0 = F.log_softmax(out.scores[0].float(), dim=-1).cpu()
    gen_token_ids = out.sequences[:, -int(max_new_tokens) :].detach().cpu().tolist()
    completions = tokenizer.batch_decode(
        gen_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    first_idents = [extract_first_identifier(text) for text in completions]
    return log_probs0, first_idents, completions


def batch_iter(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def precompute_base(
    *,
    model,
    tokenizer,
    items: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache: list[dict[str, Any]] = []
    base_errors = 0
    base_targets: list[str] = []
    base_preds: list[str] = []
    for start, batch in batch_iter(items, batch_size):
        prompts = [obj["prompt"] for obj in batch]
        targets = [obj["target_chain"] for obj in batch]
        log_probs0, first_idents, _ = run_batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )
        cache.append(
            {
                "start": start,
                "targets": targets,
                "prompts": prompts,
                "base_log_probs": log_probs0,
                "base_first_idents": first_idents,
            }
        )
        base_errors += sum(1 for pred, tgt in zip(first_idents, targets) if pred != tgt)
        base_targets.extend(targets)
        base_preds.extend(first_idents)
    return cache, {
        "base_error_rate": base_errors / max(1, len(items)),
        "base_error_count": base_errors,
        "base_target_preview": base_targets[:10],
        "base_pred_preview": base_preds[:10],
    }


def evaluate_neurons(
    *,
    model,
    tokenizer,
    base_cache: list[dict[str, Any]],
    neurons: list[tuple[int, int]],
    proj_attr: str,
    max_new_tokens: int,
    gaussian_std: float,
    gaussian_seed: int,
) -> dict[str, float]:
    zero_errors = 0
    gauss_errors = 0
    zero_kl_sum = 0.0
    gauss_kl_sum = 0.0
    total = 0

    for batch in base_cache:
        prompts = batch["prompts"]
        targets = batch["targets"]
        base_log_probs = batch["base_log_probs"]
        base_probs = base_log_probs.exp()

        hooks = register_zero_hook(model, neurons, proj_attr)
        try:
            zero_log_probs, zero_first_idents, _ = run_batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
            )
        finally:
            for hook in hooks:
                hook.remove()

        changes = add_gaussian_noise_to_columns(
            model=model,
            neurons=neurons,
            proj_attr=proj_attr,
            std=gaussian_std,
            seed=gaussian_seed + total,
        )
        try:
            gauss_log_probs, gauss_first_idents, _ = run_batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
            )
        finally:
            restore_columns(changes)

        zero_kl = (base_probs * (base_log_probs - zero_log_probs)).sum(dim=-1)
        gauss_kl = (base_probs * (base_log_probs - gauss_log_probs)).sum(dim=-1)

        zero_errors += sum(1 for pred, tgt in zip(zero_first_idents, targets) if pred != tgt)
        gauss_errors += sum(1 for pred, tgt in zip(gauss_first_idents, targets) if pred != tgt)
        zero_kl_sum += float(zero_kl.sum().item())
        gauss_kl_sum += float(gauss_kl.sum().item())
        total += len(targets)

    return {
        "n_samples": total,
        "error_rate_zero": zero_errors / max(1, total),
        "error_rate_gaussian": gauss_errors / max(1, total),
        "kl_mean_zero": zero_kl_sum / max(1, total),
        "kl_mean_gaussian": gauss_kl_sum / max(1, total),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate matrix-top200 ablation on other-torch retention prompts.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=800, help="Use at most this many retention prompts.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--random-runs", type=int, default=10)
    parser.add_argument("--gaussian-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    args = parser.parse_args()

    spec = MODEL_SPECS[str(args.model)]
    output_dir = args.output_dir / str(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(args.device))
    torch_dtype = resolve_dtype(device, str(args.dtype))
    model, tokenizer = load_model_and_tokenizer(str(spec["model_path"]), device, torch_dtype)

    try:
        items = load_dataset(Path(spec["dataset"]), limit=int(args.limit) if args.limit is not None else None)
        top_neurons = load_top_neurons(Path(spec["neuron_json"]), top_k=200)
        layers = get_model_layers(model)
        num_layers = len(layers)
        intermediate_dim = getattr(layers[0].mlp, str(spec["proj_attr"])).in_features
        if args.max_new_tokens is None:
            max_new_tokens = 8 if spec["family"] in {"deepseek-coder", "codellama"} else 3
        else:
            max_new_tokens = int(args.max_new_tokens)

        t0 = time.time()
        base_cache, base_meta = precompute_base(
            model=model,
            tokenizer=tokenizer,
            items=items,
            batch_size=int(args.batch_size),
            max_new_tokens=max_new_tokens,
        )

        top200_summary = evaluate_neurons(
            model=model,
            tokenizer=tokenizer,
            base_cache=base_cache,
            neurons=top_neurons,
            proj_attr=str(spec["proj_attr"]),
            max_new_tokens=max_new_tokens,
            gaussian_std=float(args.gaussian_std),
            gaussian_seed=int(args.seed),
        )

        random_runs = sample_layer_matched_random_runs(
            top_neurons=top_neurons,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            num_runs=int(args.random_runs),
            seed=int(args.seed),
        )
        random_summaries = []
        for run in random_runs:
            summary = evaluate_neurons(
                model=model,
                tokenizer=tokenizer,
                base_cache=base_cache,
                neurons=list(run["neurons"]),
                proj_attr=str(spec["proj_attr"]),
                max_new_tokens=max_new_tokens,
                gaussian_std=float(args.gaussian_std),
                gaussian_seed=int(run["seed"]) + 10_000,
            )
            summary["run_index"] = int(run["run_index"])
            summary["seed"] = int(run["seed"])
            random_summaries.append(summary)

        def _mean(key: str) -> float:
            return sum(float(x[key]) for x in random_summaries) / max(1, len(random_summaries))

        def _std(key: str) -> float:
            if len(random_summaries) <= 1:
                return 0.0
            mu = _mean(key)
            return (
                sum((float(x[key]) - mu) ** 2 for x in random_summaries) / len(random_summaries)
            ) ** 0.5

        summary = {
            "meta": {
                "model": args.model,
                "model_path": spec["model_path"],
                "dataset": str(spec["dataset"]),
                "neuron_json": str(spec["neuron_json"]),
                "proj_attr": spec["proj_attr"],
                "limit": args.limit,
                "used_samples": len(items),
                "batch_size": args.batch_size,
                "max_new_tokens": max_new_tokens,
                "gaussian_std": args.gaussian_std,
                "random_runs": args.random_runs,
                "random_strategy": "layer_matched_excluding_top200",
                "device": str(device),
                "dtype": str(torch_dtype),
                "elapsed_sec": time.time() - t0,
            },
            "dataset_stats": {
                "category_counts": dict(Counter(obj["category"] for obj in items)),
                "api_count": len({obj["api"] for obj in items}),
            },
            "base": base_meta,
            "top200": top200_summary,
            "random200": {
                "n_runs": len(random_summaries),
                "run_avg": {
                    "error_rate_zero": _mean("error_rate_zero"),
                    "error_rate_gaussian": _mean("error_rate_gaussian"),
                    "kl_mean_zero": _mean("kl_mean_zero"),
                    "kl_mean_gaussian": _mean("kl_mean_gaussian"),
                },
                "run_std": {
                    "error_rate_zero": _std("error_rate_zero"),
                    "error_rate_gaussian": _std("error_rate_gaussian"),
                    "kl_mean_zero": _std("kl_mean_zero"),
                    "kl_mean_gaussian": _std("kl_mean_gaussian"),
                },
                "runs": random_summaries,
            },
        }

        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        random_payload = {
            "strategy": "layer_matched_excluding_top200",
            "top_neuron_layer_counts": dict(Counter(layer for layer, _ in top_neurons)),
            "runs": [
                {
                    "run_index": int(run["run_index"]),
                    "seed": int(run["seed"]),
                    "neurons": [[int(layer), int(dim)] for layer, dim in run["neurons"]],
                }
                for run in random_runs
            ],
        }
        (output_dir / "random_layer_matched_runs.json").write_text(
            json.dumps(random_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        cleanup_model(model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
