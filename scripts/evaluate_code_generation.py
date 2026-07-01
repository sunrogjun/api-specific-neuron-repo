#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
CODEGEN_SCRIPT = Path(__file__).resolve().parents[2] / "test_generation_starcoder.py"
sys.path.insert(0, str(REPO_ROOT / "src"))

from api_neuron.local_dpo import (  # noqa: E402
    apply_edit_payload,
    apply_state_dict_payload,
    normalize_edit_mode,
    set_local_edits_enabled,
)


def load_codegen_module():
    module_name = "api_neuron_codegen_eval"
    spec = importlib.util.spec_from_file_location(module_name, CODEGEN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load code-generation helpers from {CODEGEN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(device: torch.device, dtype: str) -> torch.dtype:
    name = str(dtype).lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def infer_model_style(model_preset: str, model_path: str) -> dict[str, Any]:
    text = (model_preset or model_path or "").lower()
    if "starcoder2" in text:
        return {"use_fim": True, "use_chat": False, "proj_attr": "c_proj", "fix_indent": False}
    if "deepseek" in text:
        return {"use_fim": False, "use_chat": True, "proj_attr": "down_proj", "fix_indent": False}
    if "codellama" in text:
        return {"use_fim": False, "use_chat": False, "proj_attr": "down_proj", "fix_indent": True}
    return {"use_fim": False, "use_chat": False, "proj_attr": "down_proj", "fix_indent": False}


def build_eval_args(batch_size: int, max_new_tokens: int, eval_timeout: float | None, verbose: bool, use_stopping_criteria: bool):
    return SimpleNamespace(
        batch_size=batch_size,
        max_length_generation=None,
        max_new_tokens=max_new_tokens,
        min_new_tokens=0,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        do_sample=False,
        use_stopping_criteria=use_stopping_criteria,
        fallback_prompt="none",
        eval_timeout=eval_timeout,
        code_eval_workers=16,
        verbose=verbose,
    )


def load_tasks(codegen, task_names: list[str], num_problems: int | None, logger):
    harness_args = argparse.Namespace(prompt=None, load_data_path=None)
    tasks = []
    for name in task_names:
        task_obj = codegen.get_task(name, harness_args)
        try:
            dataset = task_obj.get_dataset()
        except AttributeError:
            codegen.ensure_task_dataset(task_obj, logger)
            dataset = task_obj.get_dataset()
        problems = list(dataset)
        if num_problems is not None:
            problems = problems[:num_problems]
        logger.log(f"Loaded {name}: {len(problems)} problems.")
        tasks.append((name, task_obj, problems))
    return tasks


def load_base_model_and_tokenizer(model_path: str, device: torch.device, torch_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)
    model.eval()
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def cleanup_model(model=None, tokenizer=None) -> None:
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


def run_pass1_eval(codegen, model, tokenizer, cfg, tasks, eval_args, prompt_mode: str, logger, output_dir: Path, mode_name: str) -> dict:
    results = {}
    for task_name, task_obj, problems in tasks:
        resolved_prompt_mode = prompt_mode
        if resolved_prompt_mode == "auto":
            resolved_prompt_mode = "instruct" if cfg.use_chat else "raw"
        save_path = output_dir / f"{mode_name}_{task_name}.jsonl"
        score = codegen.eval_model(
            model=model,
            tokenizer=tokenizer,
            proj_attr=cfg.proj_attr,
            neurons=None,
            task_name=task_name,
            task=task_obj,
            problems=problems,
            args=eval_args,
            prompt_mode=resolved_prompt_mode,
            logger=logger,
            tag=f"{mode_name}-{task_name}",
            use_fim=cfg.use_fim,
            use_chat=cfg.use_chat,
            fix_indent=cfg.fix_indent,
            save_path=save_path,
        )
        results[task_name] = {"pass@1": float(score), "n": len(problems), "prompt_mode": str(resolved_prompt_mode)}
        logger.log(f"{mode_name} {task_name} pass@1 = {score:.3f}")
    return results


def build_common_eval(args: argparse.Namespace, model_path: str, model_preset: str, output_root: Path):
    style = infer_model_style(str(model_preset), str(model_path))
    cfg = SimpleNamespace(
        name=str(model_preset),
        model_path=str(model_path),
        use_fim=bool(style["use_fim"]),
        use_chat=bool(style["use_chat"]),
        proj_attr=str(style["proj_attr"]),
        fix_indent=bool(style["fix_indent"]),
    )
    codegen = load_codegen_module()
    task_names = [item.strip().lower() for item in str(args.tasks).split(",") if item.strip()]
    logger = codegen.SimpleLogger(output_root / "codegen.log")
    output_dir = output_root / "codegen_generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = resolve_dtype(device, args.dtype)
    eval_args = build_eval_args(args.batch_size, args.max_new_tokens, args.eval_timeout, args.verbose, args.use_stopping_criteria)
    tasks = load_tasks(codegen, task_names, args.num_problems, logger)
    return codegen, cfg, logger, output_dir, device, torch_dtype, eval_args, tasks


def load_payload(adapter_path: Path) -> tuple[dict, dict]:
    payload = torch.load(adapter_path, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    return payload, meta


def apply_local_edit(model, payload: dict, meta: dict, *, device: torch.device) -> None:
    if isinstance(payload, dict) and "state_dict" in payload:
        mode_hint = normalize_edit_mode(meta.get("delta_mode")) if isinstance(meta, dict) else "full"
        apply_state_dict_payload(model, payload, mode_hint=mode_hint)
    else:
        apply_edit_payload(model, payload)
    model.to(device)
    set_local_edits_enabled(model, True)


def run_local_edit(args: argparse.Namespace) -> None:
    payload, meta = load_payload(args.adapter)
    model_path = args.model_path or (meta.get("model_path") if isinstance(meta, dict) else None)
    if not model_path:
        raise SystemExit("Could not infer model_path from adapter payload; please pass --model-path.")
    model_preset = args.model_preset or meta.get("preset") or meta.get("model_preset") or Path(str(model_path)).name
    codegen, cfg, logger, output_dir, device, torch_dtype, eval_args, tasks = build_common_eval(
        args, str(model_path), str(model_preset), args.adapter.parent
    )
    summary: dict[str, Any] = {
        "meta": {
            "adapter": str(args.adapter),
            "model_path": str(model_path),
            "model_preset": str(model_preset),
            "batch_size": int(args.batch_size),
            "max_new_tokens": int(args.max_new_tokens),
            "device": str(device),
            "dtype": str(torch_dtype),
        }
    }

    try:
        if not args.skip_base:
            base_model, base_tok = load_base_model_and_tokenizer(str(model_path), device, torch_dtype)
            summary["base"] = run_pass1_eval(codegen, base_model, base_tok, cfg, tasks, eval_args, args.prompt_mode, logger, output_dir, "base")
            cleanup_model(base_model, base_tok)

        trained_model, trained_tok = load_base_model_and_tokenizer(str(model_path), device, torch_dtype)
        apply_local_edit(trained_model, payload, meta if isinstance(meta, dict) else {}, device=device)
        summary["trained"] = run_pass1_eval(
            codegen, trained_model, trained_tok, cfg, tasks, eval_args, args.prompt_mode, logger, output_dir, "trained"
        )
        cleanup_model(trained_model, trained_tok)

        summary_path = args.adapter.parent / "codegen.summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        logger.close()


def load_run_config(adapter_dir: Path) -> dict:
    config_path = adapter_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in {adapter_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_lora_model_and_tokenizer(adapter_dir: Path, model_path: str, device: torch.device, torch_dtype: torch.dtype):
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model.eval()
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def run_lora(args: argparse.Namespace) -> None:
    run_config = load_run_config(args.adapter_dir)
    model_path = args.model_path or run_config["model_path"]
    model_preset = args.model_preset or run_config.get("model_preset") or Path(model_path).name
    codegen, cfg, logger, output_dir, device, torch_dtype, eval_args, tasks = build_common_eval(
        args, str(model_path), str(model_preset), args.adapter_dir
    )
    summary: dict[str, Any] = {
        "meta": {
            "adapter_dir": str(args.adapter_dir),
            "model_path": str(model_path),
            "model_preset": str(model_preset),
            "batch_size": int(args.batch_size),
            "max_new_tokens": int(args.max_new_tokens),
            "device": str(device),
            "dtype": str(torch_dtype),
        }
    }

    try:
        if not args.skip_base:
            base_model, base_tok = load_base_model_and_tokenizer(str(model_path), device, torch_dtype)
            summary["base"] = run_pass1_eval(codegen, base_model, base_tok, cfg, tasks, eval_args, args.prompt_mode, logger, output_dir, "base")
            cleanup_model(base_model, base_tok)

        trained_model, trained_tok = load_lora_model_and_tokenizer(args.adapter_dir, str(model_path), device, torch_dtype)
        summary["trained"] = run_pass1_eval(
            codegen, trained_model, trained_tok, cfg, tasks, eval_args, args.prompt_mode, logger, output_dir, "trained"
        )
        cleanup_model(trained_model, trained_tok)

        summary_path = args.adapter_dir / "codegen.summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        logger.close()


def add_common_args(parser: argparse.ArgumentParser, *, batch_size: int) -> None:
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-preset", default=None)
    parser.add_argument("--tasks", default="humaneval,humanevalplus")
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-timeout", type=float, default=None)
    parser.add_argument("--prompt-mode", choices=["auto", "raw", "instruct", "baseline"], default="auto")
    parser.add_argument("--use-stopping-criteria", action="store_true")
    parser.add_argument("--no-use-stopping-criteria", action="store_false", dest="use_stopping_criteria")
    parser.set_defaults(use_stopping_criteria=False)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate HumanEval/HumanEval+ code-generation ability.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local-edit", help="Evaluate a neuron-local edit payload.")
    local.add_argument("--adapter", type=Path, required=True, help="Path to local-edit payload .pt")
    add_common_args(local, batch_size=4)
    local.set_defaults(func=run_local_edit, prompt_mode="baseline")

    lora = subparsers.add_parser("lora", help="Evaluate a LoRA adapter.")
    lora.add_argument("--adapter-dir", type=Path, required=True)
    add_common_args(lora, batch_size=8)
    lora.set_defaults(func=run_lora)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
