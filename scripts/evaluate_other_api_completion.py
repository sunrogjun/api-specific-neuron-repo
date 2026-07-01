#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from api_neuron.common import (  # noqa: E402
    first_api_chain,
    get_ffn2_module,
    get_layers,
    load_model_and_tokenizer,
    read_jsonl,
    write_jsonl,
)
from api_neuron.local_dpo import (  # noqa: E402
    apply_state_dict_payload,
    normalize_edit_mode,
    set_local_edits_enabled,
)


MODEL_CONFIGS = {
    "starcoder2-3b": {
        "model_path": "/data/lkl/models/StarCoder/starcoder2-3b",
        "dpo_full": WORK_ROOT / "RL/dpo_out/neuron_deltas.pt",
        "dpo_down": WORK_ROOT / "RL/dpo_out/neuron_deltas_ffn2_weight.pt",
        "neuron_sft": REPO_ROOT / "outputs/neuron_sft/starcoder2-3b/neuron_deltas.pt",
        "lora_dpo": REPO_ROOT / "outputs/lora_dpo/starcoder2-3b/best_adapter",
    },
    "starcoder2-7b": {
        "model_path": "/data/lkl/models/StarCoder/starcoder2-7b",
        "dpo_full": WORK_ROOT / "RL/dpo_out_7b/neuron_deltas.pt",
        "dpo_down": WORK_ROOT / "RL/dpo_out_7b/neuron_deltas_ffn2_weight.pt",
        "neuron_sft": REPO_ROOT / "outputs/neuron_sft/starcoder2-7b/neuron_deltas.pt",
        "lora_dpo": REPO_ROOT / "outputs/lora_dpo/starcoder2-7b/best_adapter",
    },
    "starcoder2-15b": {
        "model_path": "/data/lkl/models/StarCoder/starcoder2-15b",
        "dpo_full": WORK_ROOT / "RL/dpo_out_15b/neuron_deltas.pt",
        "dpo_down": WORK_ROOT / "RL/dpo_out_15b/neuron_deltas_ffn2_weight.pt",
        "neuron_sft": REPO_ROOT / "outputs/neuron_sft/starcoder2-15b/neuron_deltas.pt",
        "lora_dpo": REPO_ROOT / "outputs/lora_dpo/starcoder2-15b/best_adapter",
    },
    "deepseek-coder-6.7b-instruct": {
        "model_path": "/data/lkl/models/deepseek-ai/deepseek-coder-6.7b-instruct",
        "dpo_full": WORK_ROOT / "RL/dpo_out_deepseek-6.7b/neuron_deltas.pt",
        "dpo_down": WORK_ROOT / "RL/dpo_out_deepseek-6.7b/neuron_deltas_ffn2_weight.pt",
        "neuron_sft": REPO_ROOT / "outputs/neuron_sft/deepseek-coder-6.7b-instruct/neuron_deltas.pt",
        "lora_dpo": REPO_ROOT / "outputs/lora_dpo/deepseek-coder-6.7b-instruct/best_adapter",
    },
    "codellama-7b": {
        "model_path": "/data/zzl/model/CodeLlama-7b-hf",
        "dpo_full": WORK_ROOT / "RL/dpo_out_codellama-7b/neuron_deltas.pt",
        "dpo_down": WORK_ROOT / "RL/dpo_out_codellama-7b/neuron_deltas_ffn2_weight.pt",
        "neuron_sft": REPO_ROOT / "outputs/neuron_sft/codellama-7b/neuron_deltas.pt",
        "lora_dpo": REPO_ROOT / "outputs/lora_dpo/codellama-7b/best_adapter",
    },
}

VARIANT_ORDER = ["neuron_dpo_down", "neuron_dpo_full", "neuron_sft", "lora_dpo"]


def clean_prediction(text: str) -> str:
    return first_api_chain(text)


def is_correct(predicted_chain: str, target_chain: str) -> bool:
    predicted_chain = (predicted_chain or "").strip()
    target_chain = (target_chain or "").strip()
    if not predicted_chain or not target_chain:
        return False
    if predicted_chain == target_chain:
        return True
    # Allow a model to redundantly emit "torch." after a prompt ending in
    # "torch."; first_api_chain strips it in most cases, but keep this robust.
    if predicted_chain == f"torch.{target_chain}":
        return True
    return False


def run_batched_generation(
    model,
    tokenizer,
    rows: list[dict],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    variant: str,
) -> list[dict]:
    outputs: list[dict] = []
    device = next(model.parameters()).device
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [row["prompt"] for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for offset, (row, text) in enumerate(zip(batch, texts)):
            predicted_chain = clean_prediction(text)
            outputs.append(
                {
                    "i": start + offset,
                    "id": row.get("id"),
                    "api": row["api"],
                    "target_chain": row["target_chain"],
                    "variant": variant,
                    "prediction_text": text,
                    "predicted_chain": predicted_chain,
                    "correct": is_correct(predicted_chain, row["target_chain"]),
                }
            )
    return outputs


def summarize_variant(rows: list[dict], base_rows: list[dict] | None = None) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    summary: dict[str, Any] = {
        "n": n,
        "correct": correct,
        "accuracy": round(100.0 * correct / max(1, n), 2),
    }
    if base_rows is not None:
        base_by_i = {row["i"]: row for row in base_rows}
        base_correct_items = [row for row in base_rows if row["correct"]]
        kept = 0
        changed_wrong = 0
        recovered = 0
        per_api: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "base_correct": 0, "kept": 0})
        for row in rows:
            base = base_by_i[row["i"]]
            bucket = per_api[row["api"]]
            bucket["n"] += 1
            if base["correct"]:
                bucket["base_correct"] += 1
                if row["correct"]:
                    bucket["kept"] += 1
                    kept += 1
                else:
                    changed_wrong += 1
            elif row["correct"]:
                recovered += 1
        api_retentions = [
            100.0 * item["kept"] / item["base_correct"]
            for item in per_api.values()
            if item["base_correct"] > 0
        ]
        summary.update(
            {
                "base_correct": len(base_correct_items),
                "kept_base_correct": kept,
                "changed_base_correct_to_wrong": changed_wrong,
                "recovered_base_wrong": recovered,
                "micro_retention": round(100.0 * kept / max(1, len(base_correct_items)), 2),
                "macro_api_retention": round(sum(api_retentions) / max(1, len(api_retentions)), 2),
                "apis_with_base_correct": len(api_retentions),
            }
        )
    return summary


def write_outputs(output_dir: Path, variant: str, rows: list[dict], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"{variant}.jsonl", rows)
    (output_dir / f"{variant}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class Ffn2WeightController:
    def __init__(self, model, payload: dict):
        self._enabled = False
        self._entries = []
        for record in (payload.get("ffn2_weight", {}) or {}).get("layers", []):
            layer_idx = int(record["layer"])
            cols = [int(x) for x in record["cols"]]
            delta = torch.as_tensor(record["delta"], dtype=torch.float32)
            module, _ = get_ffn2_module(get_layers(model)[layer_idx])
            base_cols = module.weight.detach().float().cpu()[:, cols].contiguous()
            if tuple(delta.shape) != tuple(base_cols.shape):
                raise ValueError(
                    f"ffn2_weight shape mismatch at layer {layer_idx}: "
                    f"delta={tuple(delta.shape)} expected={tuple(base_cols.shape)}"
                )
            self._entries.append((module.weight, cols, base_cols, delta.contiguous()))

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        for weight, cols, base_cols, delta in self._entries:
            target = base_cols + delta if enabled else base_cols
            weight.data[:, cols] = target.to(device=weight.data.device, dtype=weight.data.dtype)
        self._enabled = enabled


def load_lora_model(model_path: str, adapter_dir: Path, device: str, dtype: str):
    torch_device = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.bfloat16 if torch_device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch_device.type == "cuda" else torch.float32
        )
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(torch_device)
    model.eval()
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    trained = PeftModel.from_pretrained(model, str(adapter_dir)).to(torch_device)
    trained.eval()
    return trained, tokenizer


def evaluate_model(args) -> dict[str, Any]:
    config = MODEL_CONFIGS[args.model]
    output_dir = args.output_dir / args.model
    rows = read_jsonl(args.data_file)
    if args.limit:
        rows = rows[: args.limit]
    requested = set(VARIANT_ORDER if args.variants == "all" else [item.strip() for item in args.variants.split(",") if item.strip()])
    unknown = requested - set(VARIANT_ORDER)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")

    print(f"[info] loading base model for {args.model}: {config['model_path']}", flush=True)
    model, tokenizer, _ = load_model_and_tokenizer(config["model_path"], device=args.device, dtype=args.dtype)
    model.config.use_cache = True

    base_path = output_dir / "base.jsonl"
    base_summary_path = output_dir / "base.summary.json"
    if args.reuse_base and base_path.exists() and base_summary_path.exists():
        print(f"[info] {args.model}: reusing base completions from {base_path}", flush=True)
        base_rows = read_jsonl(base_path)
        base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    else:
        print(f"[info] {args.model}: generating base completions", flush=True)
        base_rows = run_batched_generation(
            model,
            tokenizer,
            rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            variant="base",
        )
        base_summary = summarize_variant(base_rows)
        write_outputs(output_dir, "base", base_rows, base_summary)

    summaries = {"base": base_summary}

    if "neuron_dpo_down" in requested:
        print(f"[info] {args.model}: evaluating Neuron-DPO W_down", flush=True)
        down_payload = torch.load(config["dpo_down"], map_location="cpu")
        down_controller = Ffn2WeightController(model, down_payload)
        down_controller.set_enabled(True)
        down_rows = run_batched_generation(
            model,
            tokenizer,
            rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            variant="neuron_dpo_down",
        )
        down_controller.set_enabled(False)
        summaries["neuron_dpo_down"] = summarize_variant(down_rows, base_rows)
        write_outputs(output_dir, "neuron_dpo_down", down_rows, summaries["neuron_dpo_down"])

    if "neuron_dpo_full" in requested:
        print(f"[info] {args.model}: evaluating Neuron-DPO W_up+W_down", flush=True)
        full_payload = torch.load(config["dpo_full"], map_location="cpu")
        mode_hint = normalize_edit_mode((full_payload.get("meta") or {}).get("delta_mode"))
        apply_state_dict_payload(model, full_payload, mode_hint=mode_hint)
        model = model.to(next(model.parameters()).device)
        set_local_edits_enabled(model, True)
        full_rows = run_batched_generation(
            model,
            tokenizer,
            rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            variant="neuron_dpo_full",
        )
        summaries["neuron_dpo_full"] = summarize_variant(full_rows, base_rows)
        write_outputs(output_dir, "neuron_dpo_full", full_rows, summaries["neuron_dpo_full"])

    if "neuron_sft" in requested:
        print(f"[info] {args.model}: evaluating Neuron-SFT", flush=True)
        sft_payload = torch.load(config["neuron_sft"], map_location="cpu")
        mode_hint = normalize_edit_mode((sft_payload.get("meta") or {}).get("delta_mode"))
        apply_state_dict_payload(model, sft_payload, mode_hint=mode_hint)
        model = model.to(next(model.parameters()).device)
        set_local_edits_enabled(model, True)
        sft_rows = run_batched_generation(
            model,
            tokenizer,
            rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            variant="neuron_sft",
        )
        summaries["neuron_sft"] = summarize_variant(sft_rows, base_rows)
        write_outputs(output_dir, "neuron_sft", sft_rows, summaries["neuron_sft"])

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    if "lora_dpo" in requested:
        print(f"[info] {args.model}: evaluating LoRA-DPO", flush=True)
        lora_model, lora_tokenizer = load_lora_model(
            config["model_path"],
            config["lora_dpo"],
            device=args.device,
            dtype=args.dtype,
        )
        lora_rows = run_batched_generation(
            lora_model,
            lora_tokenizer,
            rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            variant="lora_dpo",
        )
        summaries["lora_dpo"] = summarize_variant(lora_rows, base_rows)
        write_outputs(output_dir, "lora_dpo", lora_rows, summaries["lora_dpo"])

        del lora_model
        del lora_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    aggregate = {
        "model": args.model,
        "data_file": str(args.data_file),
        "num_samples": len(rows),
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate other PyTorch API completion retention after local edits.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit over input rows.")
    parser.add_argument("--variants", default="all", help="Comma-separated subset of variants, or 'all'.")
    parser.add_argument("--reuse-base", action="store_true", help="Reuse existing base.jsonl and base.summary.json if present.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    args = parser.parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
