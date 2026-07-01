#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_neuron.common import group_neurons, load_model_and_tokenizer, load_top_neurons, read_jsonl  # noqa: E402
from api_neuron.local_dpo import attach_local_edits, export_ffn2_weight_payload, export_state_dict_payload, trainable_parameters  # noqa: E402


MODEL_PATHS = {
    "starcoder2-3b": "/data/lkl/models/StarCoder/starcoder2-3b",
    "starcoder2-7b": "/data/lkl/models/StarCoder/starcoder2-7b",
    "starcoder2-15b": "/data/lkl/models/StarCoder/starcoder2-15b",
    "deepseek-coder-6.7b-instruct": "/data/lkl/models/deepseek-ai/deepseek-coder-6.7b-instruct",
    "codellama-7b": "/data/zzl/model/CodeLlama-7b-hf",
}


def build_batch(tokenizer, rows: Sequence[dict], max_length: int, max_prompt_length: int, add_eos: bool = True):
    encoded_rows = []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids[-max_prompt_length:]
        max_completion = max_length - len(prompt_ids)
        chosen_ids = tokenizer(row["chosen"], add_special_tokens=False).input_ids[:max_completion]
        sequence = prompt_ids + chosen_ids
        labels = [-100] * len(prompt_ids) + chosen_ids[:]
        if add_eos and eos_id is not None and len(sequence) < max_length:
            sequence.append(eos_id)
            labels.append(eos_id)
        encoded_rows.append((sequence, labels))

    width = max(len(sequence) for sequence, _ in encoded_rows)
    input_ids = torch.full((len(encoded_rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for idx, (sequence, label_ids) in enumerate(encoded_rows):
        input_ids[idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[idx, : len(sequence)] = 1
        labels[idx, : len(label_ids)] = torch.tensor(label_ids, dtype=torch.long)
    return input_ids, attention_mask, labels


def completion_loss(model, input_ids, attention_mask, labels):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    target = labels[:, 1:]
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=-100,
    )


def evaluate(model, loader, device, autocast_ctx):
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with autocast_ctx:
                loss = completion_loss(model, input_ids, attention_mask, labels)
            total_loss += loss.item() * input_ids.size(0)
            total_items += input_ids.size(0)
    model.train()
    return total_loss / max(1, total_items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neuron-local SFT baseline on localized FFN neurons.")
    parser.add_argument("--model-preset", required=True, choices=sorted(MODEL_PATHS))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--neuron-file", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--valid-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["down_only", "full"], default="full")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=480)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dpo_dir = repo_root / "data" / "main_case" / "dpo" / args.model_preset
    args.model_path = args.model_path or MODEL_PATHS[args.model_preset]
    args.neuron_file = args.neuron_file or (repo_root / "results" / "localization" / "matrix" / f"{args.model_preset}.top200.json")
    args.train_file = args.train_file or (dpo_dir / "train.jsonl")
    args.valid_file = args.valid_file or (dpo_dir / "valid.jsonl")
    args.output = args.output or (repo_root / "outputs" / "neuron_sft" / args.model_preset / "neuron_deltas.pt")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lr_defaults = {
        "starcoder2-3b": 5e-5,
        "starcoder2-7b": 5e-5,
        "starcoder2-15b": 3e-5,
        "deepseek-coder-6.7b-instruct": 3e-5,
        "codellama-7b": 2e-5,
    }
    args.learning_rate = args.learning_rate or lr_defaults[args.model_preset]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_rows = read_jsonl(args.train_file)
    valid_rows = read_jsonl(args.valid_file)
    if args.limit_train is not None:
        train_rows = train_rows[: args.limit_train]
    if args.limit_valid is not None:
        valid_rows = valid_rows[: args.limit_valid]

    model, tokenizer, device = load_model_and_tokenizer(args.model_path, device=args.device, dtype=args.dtype)
    model.train()
    for module in model.modules():
        if module.__class__.__name__ == "Dropout":
            module.p = 0.0

    neurons = load_top_neurons(args.neuron_file, top_k=args.top_k)
    attach_local_edits(model, group_neurons(neurons), mode=args.mode)
    model.to(device)
    if hasattr(model, "config"):
        model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = trainable_parameters(model)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if (args.dtype == "auto" and use_bf16) else (
        torch.float16 if (args.dtype == "auto" and device.type == "cuda") else getattr(torch, args.dtype)
        if args.dtype != "auto"
        else torch.float32
    )
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch_dtype) if device.type == "cuda" and torch_dtype != torch.float32 else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and torch_dtype == torch.float16))

    train_loader = DataLoader(
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda rows: build_batch(tokenizer, rows, args.max_length, args.max_prompt_length),
    )
    valid_loader = DataLoader(
        valid_rows,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda rows: build_batch(tokenizer, rows, args.max_length, args.max_prompt_length),
    )

    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))

    run_config = {
        "method": "neuron_sft",
        "model_preset": args.model_preset,
        "model_path": args.model_path,
        "neuron_file": str(args.neuron_file),
        "train_file": str(args.train_file),
        "valid_file": str(args.valid_file),
        "top_k": args.top_k,
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
        "device": str(device),
        "dtype": str(torch_dtype),
        "seed": args.seed,
    }
    (args.output.parent / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_loss = float("inf")
    best_epoch = -1
    patience = 0
    global_step = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with autocast_ctx:
                loss = completion_loss(model, input_ids, attention_mask, labels)
                scaled_loss = loss / max(1, args.grad_accum)
            epoch_losses.append(float(loss.detach().float().item()))
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if batch_idx % max(1, args.grad_accum) == 0 or batch_idx == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if global_step >= warmup_steps:
                    scheduler.step()
                global_step += 1

        valid_loss = evaluate(model, valid_loader, device=device, autocast_ctx=autocast_ctx)
        record = {
            "epoch": epoch,
            "train_loss": float(sum(epoch_losses) / max(1, len(epoch_losses))),
            "valid_loss": float(valid_loss),
            "step": global_step,
            "time": time.time(),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if valid_loss < best_loss - args.early_stop_min_delta:
            best_loss = valid_loss
            best_epoch = epoch
            patience = 0
            if args.mode == "down_only":
                payload = export_ffn2_weight_payload(
                    model,
                    model_path=args.model_path,
                    neuron_file=args.neuron_file,
                    top_k=args.top_k,
                )
            else:
                payload = export_state_dict_payload(
                    model,
                    model_path=args.model_path,
                    neuron_file=args.neuron_file,
                    top_k=args.top_k,
                    mode=args.mode,
                )
            torch.save(payload, args.output)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"[stop] early stopping at epoch {epoch}")
                break

    train_summary = {
        **run_config,
        "best_eval_loss": best_loss,
        "best_epoch": best_epoch,
        "total_optimizer_steps": global_step,
        "elapsed_sec": time.time() - t0,
    }
    (args.output.parent / "train_summary.json").write_text(json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output.with_suffix(".log.json")).write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
