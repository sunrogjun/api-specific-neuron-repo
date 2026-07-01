#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_SPECS = {
    "starcoder2-3b": {
        "model_path": "/data/lkl/models/StarCoder/starcoder2-3b",
        "proj_attr": "c_proj",
        "neuron_file": "top400/matrix/neuron_top_starcoder2-3b.json",
        "dataset": "matrix_v2/sc-3B/matrix_dataset_v2_starcoder2-3b_base_matches.jsonl",
    },
    "starcoder2-7b": {
        "model_path": "/data/lkl/models/StarCoder/starcoder2-7b",
        "proj_attr": "c_proj",
        "neuron_file": "top400/matrix/neuron_top_starcoder2-7b.json",
        "dataset": "matrix_v2/sc-7B/matrix_dataset_v2_starcoder2-7b_base_matches.jsonl",
    },
    "starcoder2-15b": {
        "model_path": "/data/lkl/models/StarCoder/starcoder2-15b",
        "proj_attr": "c_proj",
        "neuron_file": "top400/matrix/neuron_top_starcoder2-15b.json",
        "dataset": "matrix_v2/sc-15B/matrix_dataset_v2_starcoder2-15b_base_matches.jsonl",
    },
    "deepseek-coder-6.7b-instruct": {
        "model_path": "/data/lkl/models/deepseek-ai/deepseek-coder-6.7b-instruct",
        "proj_attr": "down_proj",
        "neuron_file": "top400/matrix/neuron_top_deepseek-coder-6.7b-instruct.json",
        "dataset": "matrix_v2/deepseek-6.7B/matrix_dataset_v2_deepseek-coder-6.7b-instruct_base_matches.jsonl",
    },
    "codellama-7b": {
        "model_path": "/data/zzl/model/CodeLlama-7b-hf",
        "proj_attr": "down_proj",
        "neuron_file": "top400/matrix/neuron_top_CodeLlama-7b-hf.json",
        "dataset": "matrix_v2/codellama-7b/matrix_dataset_v2_CodeLlama-7b-hf_base_matches.jsonl",
    },
}


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_top_neurons(path: Path, top_k: int) -> List[Tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    neurons = []
    seen = set()
    for item in data.get("global_top_neurons", []):
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


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported transformer layout.")


def group_neurons(neurons: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = {}
    for layer, neuron in neurons:
        grouped.setdefault(int(layer), []).append(int(neuron))
    return {layer: sorted(set(indices)) for layer, indices in grouped.items()}


def register_zero_hooks(model, proj_attr: str, neurons: Sequence[Tuple[int, int]]):
    hooks = []
    by_layer = group_neurons(neurons)
    for layer_idx, dims in by_layer.items():
        proj = getattr(get_layers(model)[layer_idx].mlp, proj_attr)

        def hook(_module, inputs, dims=tuple(dims)):
            hidden = inputs[0].clone()
            hidden[..., list(dims)] = 0
            return (hidden,)

        hooks.append(proj.register_forward_pre_hook(hook))
    return hooks


def add_gaussian_noise(model, proj_attr: str, neurons: Sequence[Tuple[int, int]], std: float, seed: int):
    generator = torch.Generator(device=next(model.parameters()).device)
    generator.manual_seed(int(seed))
    backups = []
    with torch.no_grad():
        for layer_idx, dims in group_neurons(neurons).items():
            weight = getattr(get_layers(model)[layer_idx].mlp, proj_attr).weight
            cols = torch.tensor(sorted(dims), device=weight.device)
            original = weight[:, cols].clone()
            noise = torch.randn(
                original.shape,
                generator=generator,
                device=weight.device,
                dtype=weight.dtype,
            ) * float(std)
            weight[:, cols] = original + noise
            backups.append((weight, cols, original))
    return backups


def restore_columns(backups):
    with torch.no_grad():
        for weight, cols, original in backups:
            weight[:, cols] = original


def first_identifier(text: str) -> str:
    token = []
    for char in (text or "").lstrip():
        if char.isalnum() or char == "_":
            token.append(char)
        elif token:
            break
    return "".join(token)


def clean_token(token: str) -> str:
    while token.startswith(("Ġ", "▁", " ")):
        token = token[1:]
    return token


def batch_generate_scores(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, List[str], List[List[Tuple[str, float]]]]:
    device = next(model.parameters()).device
    all_log_probs = []
    all_first = []
    all_top5 = []
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx is None or max_ctx > 1_000_000:
        max_ctx = getattr(model.config, "n_positions", 2048)
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_ctx),
            return_token_type_ids=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        if not output.scores:
            raise RuntimeError("Generation did not return scores.")
        log_probs0 = F.log_softmax(output.scores[0].float(), dim=-1).cpu()
        all_log_probs.append(log_probs0)
        gen_token_ids = output.sequences[:, -int(max_new_tokens) :].detach().cpu().tolist()
        completions = tokenizer.batch_decode(
            gen_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        all_first.extend(first_identifier(text) for text in completions)
        probs = log_probs0.exp()
        values, indices = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)
        for row_values, row_indices in zip(values.tolist(), indices.tolist()):
            all_top5.append(
                [
                    (clean_token(tokenizer.convert_ids_to_tokens([int(idx)])[0]), round(float(value), 6))
                    for value, idx in zip(row_values, row_indices)
                ]
            )
    return torch.cat(all_log_probs, dim=0), all_first, all_top5


def kl_from_log_probs(base_log_probs: torch.Tensor, edit_log_probs: torch.Tensor) -> torch.Tensor:
    return (base_log_probs.exp() * (base_log_probs - edit_log_probs)).sum(dim=-1)


def summarize_intervention(
    *,
    model,
    tokenizer,
    prompts: Sequence[str],
    targets: Sequence[str],
    base_log_probs: torch.Tensor,
    base_first: Sequence[str],
    neurons: Sequence[Tuple[int, int]],
    proj_attr: str,
    mode: str,
    batch_size: int,
    max_new_tokens: int,
    noise_std: float,
    seed: int,
) -> dict:
    with ExitStack() as stack:
        if mode == "zero":
            for hook in register_zero_hooks(model, proj_attr, neurons):
                stack.callback(hook.remove)
        elif mode == "gaussian":
            backups = add_gaussian_noise(model, proj_attr, neurons, std=noise_std, seed=seed)
            stack.callback(lambda: restore_columns(backups))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        edit_log_probs, edit_first, top5 = batch_generate_scores(
            model,
            tokenizer,
            prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

    kl = kl_from_log_probs(base_log_probs, edit_log_probs)
    target_errors = [pred != target for pred, target in zip(edit_first, targets)]
    base_shift = [pred != base for pred, base in zip(edit_first, base_first)]
    return {
        "n": len(prompts),
        "target_error_rate": sum(target_errors) / max(1, len(target_errors)),
        "base_shift_rate": sum(base_shift) / max(1, len(base_shift)),
        "kl_mean": float(kl.mean().item()),
        "kl_median": float(kl.median().item()),
        "first_predictions": edit_first,
        "top5": top5,
    }


def sample_random_neurons(
    *,
    model,
    proj_attr: str,
    count: int,
    seed: int,
    exclude: set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    layers = get_layers(model)
    intermediate_dim = getattr(layers[0].mlp, proj_attr).in_features
    selected = []
    selected_set = set()
    while len(selected) < count:
        pair = (rng.randrange(0, len(layers)), rng.randrange(0, intermediate_dim))
        if pair in exclude or pair in selected_set:
            continue
        selected.append(pair)
        selected_set.add(pair)
    return selected


def strip_large_fields(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in {"first_predictions", "top5"}}


def aggregate_random(runs: Iterable[dict]) -> dict:
    runs = list(runs)
    output = {"n": runs[0]["n"] if runs else 0}
    for key in ["target_error_rate", "base_shift_rate", "kl_mean", "kl_median"]:
        values = [float(run[key]) for run in runs]
        output[key] = mean(values) if values else math.nan
        output[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return output


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Matrix-prompt ablation for arbitrary top-K localized neurons.")
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--random-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--save-details", action="store_true")
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model_key]
    repo_root = args.repo_root.resolve()
    dataset_path = repo_root / spec["dataset"]
    neuron_path = repo_root / spec["neuron_file"]
    rows = read_jsonl(dataset_path)
    prompts = [str(row["prompt"]) for row in rows]
    targets = [str(row.get("ground_truth_0") or row.get("pred_token") or "").strip() for row in rows]
    if not all(targets):
        raise ValueError(f"Missing target field in {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(spec["model_path"], trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_path"],
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    top_neurons = load_top_neurons(neuron_path, args.top_k)
    exclude = set(load_top_neurons(neuron_path, max(args.top_k, 200)))

    base_log_probs, base_first, base_top5 = batch_generate_scores(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    base_acc = sum(pred == target for pred, target in zip(base_first, targets)) / max(1, len(targets))

    print(f"[{args.model_key} top{args.top_k}] n={len(prompts)} base_acc={base_acc:.4f}")
    top_zero = summarize_intervention(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        targets=targets,
        base_log_probs=base_log_probs,
        base_first=base_first,
        neurons=top_neurons,
        proj_attr=spec["proj_attr"],
        mode="zero",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    print(
        f"[{args.model_key} top{args.top_k}] top zero err={top_zero['target_error_rate']:.4f} "
        f"shift={top_zero['base_shift_rate']:.4f} KL={top_zero['kl_mean']:.4f}"
    )
    top_gaussian = summarize_intervention(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        targets=targets,
        base_log_probs=base_log_probs,
        base_first=base_first,
        neurons=top_neurons,
        proj_attr=spec["proj_attr"],
        mode="gaussian",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        noise_std=args.noise_std,
        seed=args.seed + 1000,
    )
    print(
        f"[{args.model_key} top{args.top_k}] top gaussian err={top_gaussian['target_error_rate']:.4f} "
        f"shift={top_gaussian['base_shift_rate']:.4f} KL={top_gaussian['kl_mean']:.4f}"
    )

    random_zero_runs = []
    random_gaussian_runs = []
    random_neuron_runs = []
    for run_index in range(1, args.random_runs + 1):
        run_seed = args.seed + run_index - 1
        neurons = sample_random_neurons(
            model=model,
            proj_attr=spec["proj_attr"],
            count=args.top_k,
            seed=run_seed,
            exclude=exclude,
        )
        random_neuron_runs.append({"run_index": run_index, "seed": run_seed, "neurons": neurons})
        random_zero = summarize_intervention(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            targets=targets,
            base_log_probs=base_log_probs,
            base_first=base_first,
            neurons=neurons,
            proj_attr=spec["proj_attr"],
            mode="zero",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            noise_std=args.noise_std,
            seed=run_seed,
        )
        random_gaussian = summarize_intervention(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            targets=targets,
            base_log_probs=base_log_probs,
            base_first=base_first,
            neurons=neurons,
            proj_attr=spec["proj_attr"],
            mode="gaussian",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            noise_std=args.noise_std,
            seed=run_seed + 1000,
        )
        random_zero_runs.append(strip_large_fields(random_zero))
        random_gaussian_runs.append(strip_large_fields(random_gaussian))
        print(
            f"[{args.model_key} top{args.top_k}] random {run_index}/{args.random_runs} "
            f"zero_err={random_zero['target_error_rate']:.4f} gauss_err={random_gaussian['target_error_rate']:.4f}"
        )

    payload = {
        "model": args.model_key,
        "top_k": args.top_k,
        "dataset": str(dataset_path),
        "neuron_file": str(neuron_path),
        "n": len(prompts),
        "target_distribution": dict(Counter(targets)),
        "base": {
            "target_accuracy": base_acc,
            "first_predictions": base_first if args.save_details else None,
            "top5": base_top5 if args.save_details else None,
        },
        "localized": {
            "zero": top_zero if args.save_details else strip_large_fields(top_zero),
            "gaussian": top_gaussian if args.save_details else strip_large_fields(top_gaussian),
        },
        "random": {
            "num_runs": args.random_runs,
            "zero_mean": aggregate_random(random_zero_runs),
            "gaussian_mean": aggregate_random(random_gaussian_runs),
            "zero_runs": random_zero_runs,
            "gaussian_runs": random_gaussian_runs,
            "neuron_runs": random_neuron_runs,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.model_key} top{args.top_k}] wrote {args.output}")


if __name__ == "__main__":
    main()
