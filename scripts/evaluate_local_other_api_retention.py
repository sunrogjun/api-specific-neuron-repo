#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_neuron.common import (  # noqa: E402
    first_api_chain,
    get_ffn2_module,
    get_layers,
    load_model_and_tokenizer,
    read_jsonl,
    write_jsonl,
)
from api_neuron.local_dpo import apply_edit_payload, apply_state_dict_payload, normalize_edit_mode, set_local_edits_enabled  # noqa: E402


def clean_prediction(text: str) -> str:
    return first_api_chain(text)


def is_correct(predicted_chain: str, target_chain: str) -> bool:
    predicted_chain = (predicted_chain or "").strip()
    target_chain = (target_chain or "").strip()
    if not predicted_chain or not target_chain:
        return False
    if predicted_chain == target_chain:
        return True
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate other-torch API retention for a saved local DPO adapter.")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.adapter, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    model_path = args.model_path or meta.get("model_path")
    if not model_path:
        raise SystemExit("Could not infer model_path from adapter payload; please pass --model-path.")

    rows = read_jsonl(args.data_file)
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer, _ = load_model_and_tokenizer(model_path, device=args.device, dtype=args.dtype)
    model.config.use_cache = True

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

    if isinstance(payload, dict) and "state_dict" in payload:
        mode_hint = normalize_edit_mode(meta.get("delta_mode")) if isinstance(meta, dict) else "full"
        apply_state_dict_payload(model, payload, mode_hint=mode_hint)
    elif isinstance(payload, dict) and "ffn2_weight" in payload:
        layers_payload = (payload.get("ffn2_weight", {}) or {}).get("layers", [])
        layers = get_layers(model)
        for record in layers_payload:
            layer_idx = int(record["layer"])
            cols = [int(x) for x in record["cols"]]
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(f"Layer index out of range in ffn2_weight payload: {layer_idx}")
            module, _ = get_ffn2_module(layers[layer_idx])
            delta = torch.as_tensor(record["delta"], dtype=torch.float32)
            base_cols = module.weight.detach().float().cpu()[:, cols].contiguous()
            if tuple(delta.shape) != tuple(base_cols.shape):
                raise ValueError(
                    f"ffn2_weight shape mismatch at layer {layer_idx}: "
                    f"delta={tuple(delta.shape)} expected={tuple(base_cols.shape)}"
                )
            module.weight.data[:, cols] = (base_cols + delta).to(device=module.weight.device, dtype=module.weight.dtype)
    else:
        apply_edit_payload(model, payload)
    model = model.to(next(model.parameters()).device)
    set_local_edits_enabled(model, True)

    trained_rows = run_batched_generation(
        model,
        tokenizer,
        rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        variant="trained",
    )
    trained_summary = summarize_variant(trained_rows, base_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "base.jsonl", base_rows)
    write_jsonl(args.output_dir / "trained.jsonl", trained_rows)
    aggregate = {
        "adapter": str(args.adapter),
        "data_file": str(args.data_file),
        "num_samples": len(rows),
        "summaries": {
            "base": base_summary,
            "trained": trained_summary,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
