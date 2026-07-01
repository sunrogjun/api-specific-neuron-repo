#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from api_neuron.common import (  # noqa: E402
    api_matches,
    first_api_chain,
    get_ffn2_module,
    get_layers,
    load_model_and_tokenizer,
    read_jsonl,
    write_jsonl,
)
from api_neuron.local_dpo import (  # noqa: E402
    apply_edit_payload,
    apply_state_dict_payload,
    normalize_edit_mode,
    set_local_edits_enabled,
)


def summarize(rows: list[dict]) -> dict:
    counts = {"chosen": 0, "rejected": 0, "other": 0}
    for row in rows:
        counts[row["pred_label"]] += 1
    n = len(rows)
    return {
        "n": n,
        "chosen_preds": counts["chosen"],
        "rejected_preds": counts["rejected"],
        "other_preds": counts["other"],
        "chosen_rate": round(100.0 * counts["chosen"] / max(1, n), 1),
        "rejected_rate": round(100.0 * counts["rejected"] / max(1, n), 1),
        "other_rate": round(100.0 * counts["other"] / max(1, n), 1),
    }


def classify_local_prediction(predicted: str, row: dict) -> str:
    if api_matches(predicted, row["chosen"]):
        return "chosen"
    if api_matches(predicted, row["rejected"]):
        return "rejected"
    return "other"


def classify_lora_prediction(predicted: str, row: dict) -> str:
    if predicted == row["chosen"] or predicted.endswith("." + row["chosen"]):
        return "chosen"
    if predicted == row["rejected"] or predicted.endswith("." + row["rejected"]):
        return "rejected"
    return "other"


def run_generation(model, tokenizer, rows: list[dict], max_new_tokens: int, classifier) -> list[dict]:
    outputs = []
    for idx, row in enumerate(rows):
        encoded = tokenizer(row["prompt"], return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        generated_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        predicted_api = first_api_chain(generated_text)
        outputs.append(
            {
                "i": idx,
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "prediction_text": generated_text,
                "predicted_api": predicted_api,
                "pred_label": classifier(predicted_api, row),
            }
        )
    return outputs


def write_outputs(args: argparse.Namespace, summary: dict, trained_rows: list[dict], base_rows: list[dict] | None = None) -> None:
    if base_rows is not None:
        write_jsonl(args.output.with_suffix(".base.jsonl"), base_rows)
        summary["base"] = summarize(base_rows)
    write_jsonl(args.output, trained_rows)
    summary["trained"] = summarize(trained_rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def infer_mode_from_adapter_path(path: Path) -> str:
    name = path.stem.lower()
    if "down_only" in name or "ffn2_weight" in name:
        return "down_only"
    return "full"


def install_ffn2_weight_controller(model, payload: dict):
    class Ffn2WeightController:
        def __init__(self, model_, layers_payload):
            self._enabled = False
            self._entries = []
            for record in layers_payload:
                layer_idx = int(record["layer"])
                cols = [int(x) for x in record["cols"]]
                delta = torch.as_tensor(record["delta"], dtype=torch.float32)
                layers = get_layers(model_)
                if layer_idx < 0 or layer_idx >= len(layers):
                    raise ValueError(f"Layer index out of range in ffn2_weight payload: {layer_idx}")
                module, _ = get_ffn2_module(layers[layer_idx])
                base_cols = module.weight.detach().float().cpu()[:, cols].contiguous()
                if tuple(delta.shape) != tuple(base_cols.shape):
                    raise ValueError(
                        f"ffn2_weight shape mismatch at layer {layer_idx}: "
                        f"delta={tuple(delta.shape)} expected={tuple(base_cols.shape)}"
                    )
                self._entries.append((module.weight, cols, base_cols, delta.contiguous()))
            self.set_enabled(True)

        def set_enabled(self, enabled: bool) -> None:
            enabled = bool(enabled)
            if enabled == self._enabled:
                return
            for weight, cols, base_cols, delta in self._entries:
                target = base_cols + delta if enabled else base_cols
                weight.data[:, cols] = target.to(device=weight.data.device, dtype=weight.data.dtype)
            self._enabled = enabled

    layers_payload = (payload.get("ffn2_weight", {}) or {}).get("layers", [])
    return Ffn2WeightController(model, layers_payload)


def run_local_dpo(args: argparse.Namespace) -> None:
    payload = torch.load(args.adapter, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}

    if isinstance(payload, dict) and ("state_dict" in payload or "ffn2_weight" in payload):
        model_path = args.model_path or meta.get("model_path")
    else:
        model_path = args.model_path or payload["model_path"]
    if not model_path:
        raise SystemExit("--model-path is required because the adapter checkpoint does not store a portable model id.")

    rows = read_jsonl(args.data_file)
    model, tokenizer, _ = load_model_and_tokenizer(model_path, device=args.device, dtype=args.dtype)
    weight_controller = None

    if isinstance(payload, dict) and "state_dict" in payload:
        mode_hint = normalize_edit_mode(meta.get("delta_mode")) if isinstance(meta, dict) else None
        if not mode_hint or mode_hint == "full":
            mode_hint = infer_mode_from_adapter_path(args.adapter)
        mode, missing, unexpected = apply_state_dict_payload(model, payload, mode_hint=mode_hint)
        model = model.to(next(model.parameters()).device)
        if unexpected:
            print(f"[warn] unexpected adapter keys: {len(unexpected)}")
        if missing:
            print(f"[info] missing base-model keys during adapter load: {len(missing)}")
        print(f"[info] loaded state_dict adapter in mode={mode}")
    elif isinstance(payload, dict) and "ffn2_weight" in payload:
        weight_controller = install_ffn2_weight_controller(model, payload)
        print("[info] loaded ffn2_weight adapter in mode=down_only")
    else:
        apply_edit_payload(model, payload)
        model = model.to(next(model.parameters()).device)

    def set_adapter_enabled(enabled: bool) -> None:
        if weight_controller is not None:
            weight_controller.set_enabled(enabled)
        else:
            set_local_edits_enabled(model, enabled)

    base_rows = None
    if args.compare_base:
        set_adapter_enabled(False)
        base_rows = run_generation(model, tokenizer, rows, args.max_new_tokens, classify_local_prediction)

    set_adapter_enabled(True)
    trained_rows = run_generation(model, tokenizer, rows, args.max_new_tokens, classify_local_prediction)
    write_outputs(args, {}, trained_rows, base_rows)


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
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


def load_run_config(adapter_dir: Path) -> dict:
    for candidate in [adapter_dir / "run_config.json", adapter_dir.parent / "run_config.json"]:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing run_config.json for adapter dir: {adapter_dir}")


def load_lora_model_and_tokenizer(model_path: str, adapter_dir: Path, device: torch.device, torch_dtype: torch.dtype):
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch_dtype).to(device)
    model.eval()
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    trained = PeftModel.from_pretrained(model, str(adapter_dir)).to(device)
    trained.eval()
    return model, trained, tokenizer


def run_lora_dpo(args: argparse.Namespace) -> None:
    run_config = load_run_config(args.adapter_dir)
    model_path = args.model_path or run_config["model_path"]
    rows = read_jsonl(args.data_file)
    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(device, args.dtype)
    base_model, trained_model, tokenizer = load_lora_model_and_tokenizer(model_path, args.adapter_dir, device, torch_dtype)

    base_rows = None
    if args.compare_base:
        base_rows = run_generation(base_model, tokenizer, rows, args.max_new_tokens, classify_lora_prediction)

    trained_rows = run_generation(trained_model, tokenizer, rows, args.max_new_tokens, classify_lora_prediction)
    write_outputs(args, {}, trained_rows, base_rows)


def add_support_args(parser: argparse.ArgumentParser, *, local: bool) -> None:
    if local:
        parser.add_argument("--adapter", type=Path, required=True)
    else:
        parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--compare-base", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate API-completion mitigation on chosen/rejected support prompts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local-dpo", help="Evaluate a neuron-local DPO adapter.")
    add_support_args(local, local=True)
    local.set_defaults(func=run_local_dpo)

    lora = subparsers.add_parser("lora-dpo", help="Evaluate a layer-localized LoRA-DPO adapter.")
    add_support_args(lora, local=False)
    lora.set_defaults(func=run_lora_dpo)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
