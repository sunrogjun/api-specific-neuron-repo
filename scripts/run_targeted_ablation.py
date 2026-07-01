#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_neuron.common import (  # noqa: E402
    first_identifier,
    get_ffn2_module,
    get_layers,
    load_model_and_tokenizer,
    load_top_neurons,
    read_jsonl,
)


def target_string(row: dict) -> str:
    if isinstance(row.get("ground_truth"), list) and row["ground_truth"]:
        return str(row["ground_truth"][0]).strip()
    if isinstance(row.get("chosen"), str):
        return row["chosen"].split(".", 1)[0]
    raise ValueError("Could not infer target string from row.")


def grouped(neurons: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    output: Dict[int, List[int]] = {}
    for layer, neuron in neurons:
        output.setdefault(layer, []).append(neuron)
    return {layer: sorted(set(indices)) for layer, indices in output.items()}


def register_zero_hooks(model, neurons_by_layer: Dict[int, List[int]]):
    hooks = []
    for layer_idx, dims in neurons_by_layer.items():
        module, _ = get_ffn2_module(get_layers(model)[layer_idx])

        def hook(_module, inputs, dims=tuple(dims)):
            hidden = inputs[0].clone()
            hidden[..., list(dims)] = 0
            return (hidden,)

        hooks.append(module.register_forward_pre_hook(hook))
    return hooks


def perturb_columns(model, neurons_by_layer: Dict[int, List[int]], std: float, seed: int):
    generator = torch.Generator(device=next(model.parameters()).device)
    generator.manual_seed(seed)
    backups = []
    with torch.no_grad():
        for layer_idx, dims in neurons_by_layer.items():
            module, _ = get_ffn2_module(get_layers(model)[layer_idx])
            cols = torch.tensor(sorted(dims), device=module.weight.device)
            original = module.weight[:, cols].clone()
            noise = torch.randn(
                original.shape,
                generator=generator,
                device=module.weight.device,
                dtype=module.weight.dtype,
            ) * std
            module.weight[:, cols] = original + noise
            backups.append((module.weight, cols, original))
    return backups


def restore_columns(backups):
    with torch.no_grad():
        for weight, cols, original in backups:
            weight[:, cols] = original


def sample_random_neurons(model, count: int, seed: int) -> List[Tuple[int, int]]:
    all_neurons: List[Tuple[int, int]] = []
    for layer_idx, layer in enumerate(get_layers(model)):
        width = get_ffn2_module(layer)[0].weight.shape[1]
        all_neurons.extend((layer_idx, neuron_idx) for neuron_idx in range(width))
    rng = random.Random(seed)
    return rng.sample(all_neurons, count)


def batch_logits(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    batch_size: int,
) -> Tuple[torch.Tensor, List[str]]:
    logits_chunks = []
    predictions = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            last = encoded["attention_mask"].sum(dim=1) - 1
            logits = output.logits[torch.arange(output.logits.size(0), device=last.device), last]
            generated = model.generate(
                **encoded,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        logits_chunks.append(logits.float().cpu())
        predictions.extend(first_identifier(text) for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return torch.cat(logits_chunks, dim=0), predictions


def kl_divergence(base_logits: torch.Tensor, ablated_logits: torch.Tensor) -> torch.Tensor:
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)
    base_probs = base_log_probs.exp()
    return (base_probs * (base_log_probs - ablated_log_probs)).sum(dim=-1)


def evaluate_once(
    model,
    tokenizer,
    prompts: Sequence[str],
    targets: Sequence[str],
    neurons: Sequence[Tuple[int, int]],
    mode: str,
    max_length: int,
    batch_size: int,
    noise_std: float,
    seed: int,
) -> dict:
    base_logits, _ = batch_logits(model, tokenizer, prompts, max_length=max_length, batch_size=batch_size)
    neurons_by_layer = grouped(neurons)

    with ExitStack() as stack:
        if mode == "zero":
            for hook in register_zero_hooks(model, neurons_by_layer):
                stack.callback(hook.remove)
        elif mode == "gaussian":
            backups = perturb_columns(model, neurons_by_layer, std=noise_std, seed=seed)
            stack.callback(lambda: restore_columns(backups))

        ablated_logits, predictions = batch_logits(model, tokenizer, prompts, max_length=max_length, batch_size=batch_size)

    errors = sum(pred != target for pred, target in zip(predictions, targets))
    kl = kl_divergence(base_logits, ablated_logits)
    return {
        "n": len(prompts),
        "error_rate": errors / max(1, len(prompts)),
        "kl_mean": float(kl.mean().item()),
        "kl_median": float(kl.median().item()),
    }


def mean_metric(results: List[dict], key: str) -> float:
    return sum(result[key] for result in results) / max(1, len(results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-out or Gaussian ablation on a localized neuron set.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--neuron-file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=["zero", "gaussian"], default="gaussian")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--random-runs", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    prompts = [row["prompt"] for row in rows]
    targets = [target_string(row) for row in rows]
    model, tokenizer, _ = load_model_and_tokenizer(args.model_path, device=args.device, dtype=args.dtype)
    neurons = load_top_neurons(args.neuron_file, top_k=args.top_k)

    top_result = evaluate_once(
        model,
        tokenizer,
        prompts,
        targets,
        neurons,
        mode=args.mode,
        max_length=args.max_length,
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        seed=0,
    )

    random_results = []
    for run_idx in range(args.random_runs):
        random_neurons = sample_random_neurons(model, len(neurons), seed=run_idx)
        random_results.append(
            evaluate_once(
                model,
                tokenizer,
                prompts,
                targets,
                random_neurons,
                mode=args.mode,
                max_length=args.max_length,
                batch_size=args.batch_size,
                noise_std=args.noise_std,
                seed=run_idx + 1000,
            )
        )

    payload = {
        "dataset": str(args.dataset),
        "mode": args.mode,
        "top_k": args.top_k,
        "top_neurons": top_result,
        "random_mean": {
            "n": top_result["n"],
            "error_rate": mean_metric(random_results, "error_rate"),
            "kl_mean": mean_metric(random_results, "kl_mean"),
            "kl_median": mean_metric(random_results, "kl_median"),
        },
        "random_runs": random_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
