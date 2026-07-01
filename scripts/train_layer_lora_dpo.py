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
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_neuron.common import load_model_and_tokenizer, read_jsonl  # noqa: E402
from api_neuron.lora_sft import (  # noqa: E402
    attach_layer_lora,
    build_scheduler,
    count_trainable_parameters,
    select_top_layers,
)


MODEL_PATHS = {
    "starcoder2-3b": "/data/lkl/models/StarCoder/starcoder2-3b",
    "starcoder2-7b": "/data/lkl/models/StarCoder/starcoder2-7b",
    "starcoder2-15b": "/data/lkl/models/StarCoder/starcoder2-15b",
    "deepseek-coder-6.7b-instruct": "/data/lkl/models/deepseek-ai/deepseek-coder-6.7b-instruct",
    "codellama-7b": "/data/zzl/model/CodeLlama-7b-hf",
}


def build_batch(tokenizer, rows: Sequence[dict], max_length: int, max_prompt_length: int):
    chosen, rejected, prompt_lengths = [], [], []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids[-max_prompt_length:]
        max_completion = max_length - len(prompt_ids)
        chosen_ids = tokenizer(row["chosen"], add_special_tokens=False).input_ids[:max_completion]
        rejected_ids = tokenizer(row["rejected"], add_special_tokens=False).input_ids[:max_completion]
        chosen.append(prompt_ids + chosen_ids)
        rejected.append(prompt_ids + rejected_ids)
        prompt_lengths.append(len(prompt_ids))

    sequences = chosen + rejected
    prompt_lengths = prompt_lengths + prompt_lengths
    pad_id = tokenizer.pad_token_id
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for idx, sequence in enumerate(sequences):
        input_ids[idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[idx, : len(sequence)] = 1
    return input_ids, attention_mask, torch.tensor(prompt_lengths), len(rows)


def sequence_logps(model, input_ids, attention_mask, prompt_lengths):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    labels = input_ids[:, 1:]
    token_logps = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    mask = (positions >= (prompt_lengths.unsqueeze(1) - 1)) & attention_mask[:, 1:].bool()
    return (token_logps * mask).sum(dim=1)


def dpo_loss(policy_logps, reference_logps, batch_size: int, beta: float):
    chosen_policy = policy_logps[:batch_size]
    rejected_policy = policy_logps[batch_size:]
    chosen_reference = reference_logps[:batch_size]
    rejected_reference = reference_logps[batch_size:]
    logits = beta * ((chosen_policy - rejected_policy) - (chosen_reference - rejected_reference))
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    acc = float((logits > 0).float().mean().item())
    return loss, acc


def evaluate(model, loader, device, beta: float, autocast_ctx):
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_acc = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, prompt_lengths, batch_size in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            prompt_lengths = prompt_lengths.to(device)
            with model.disable_adapter():
                with autocast_ctx:
                    reference = sequence_logps(model, input_ids, attention_mask, prompt_lengths)
            with autocast_ctx:
                policy = sequence_logps(model, input_ids, attention_mask, prompt_lengths)
                loss, acc = dpo_loss(policy, reference, batch_size=batch_size, beta=beta)
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            total_items += batch_size
    model.train()
    return total_loss / max(1, total_items), total_acc / max(1, total_items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a layer-localized LoRA-DPO baseline on the top-3 neuron-dense layers.")
    parser.add_argument("--model-preset", required=True, choices=sorted(MODEL_PATHS))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--valid-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--layer-counts-file", type=Path, default=None)
    parser.add_argument("--seed-name", default="matrix")
    parser.add_argument("--top-layer-count", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
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
    args.model_path = args.model_path or MODEL_PATHS[args.model_preset]
    args.layer_counts_file = args.layer_counts_file or (repo_root / "results" / "localization" / "top200_layer_counts.csv")
    dpo_dir = repo_root / "data" / "main_case" / "dpo" / args.model_preset
    args.train_file = args.train_file or (dpo_dir / "train.jsonl")
    args.valid_file = args.valid_file or (dpo_dir / "valid.jsonl")
    args.output_dir = args.output_dir or (repo_root / "outputs" / "lora_dpo" / args.model_preset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    selected_layers = select_top_layers(
        layer_counts_file=args.layer_counts_file,
        model_preset=args.model_preset,
        seed=args.seed_name,
        top_n=args.top_layer_count,
    )
    model, lora_config = attach_layer_lora(
        model,
        layers_to_transform=selected_layers,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)

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

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    trainable_count, total_count = count_trainable_parameters(model)
    run_config = {
        "method": "lora_dpo",
        "model_preset": args.model_preset,
        "model_path": args.model_path,
        "train_file": str(args.train_file),
        "valid_file": str(args.valid_file),
        "selected_layers": selected_layers,
        "seed_name": args.seed_name,
        "top_layer_count": args.top_layer_count,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "beta": args.beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum,
        "max_grad_norm": args.max_grad_norm,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
        "device": str(device),
        "dtype": str(torch_dtype),
        "trainable_params": trainable_count,
        "total_params": total_count,
        "lora_target_modules": list(lora_config.target_modules or []),
        "seed": args.seed,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_dir = args.output_dir / "best_adapter"
    best_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = args.output_dir / "loss_log.jsonl"
    best_eval = float("inf")
    best_epoch = -1
    patience = 0
    global_step = 0
    t0 = time.time()

    with loss_log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            epoch_losses = []
            epoch_accs = []
            for batch_idx, (input_ids, attention_mask, prompt_lengths, batch_size) in enumerate(train_loader, start=1):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                prompt_lengths = prompt_lengths.to(device)

                with torch.no_grad():
                    with model.disable_adapter():
                        with autocast_ctx:
                            reference = sequence_logps(model, input_ids, attention_mask, prompt_lengths)
                with autocast_ctx:
                    policy = sequence_logps(model, input_ids, attention_mask, prompt_lengths)
                    loss, acc = dpo_loss(policy, reference, batch_size=batch_size, beta=args.beta)
                    scaled_loss = loss / max(1, args.grad_accum)

                epoch_losses.append(float(loss.detach().float().item()))
                epoch_accs.append(acc)
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if batch_idx % max(1, args.grad_accum) == 0 or batch_idx == len(train_loader):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            valid_loss, valid_acc = evaluate(model, valid_loader, device=device, beta=args.beta, autocast_ctx=autocast_ctx)
            record = {
                "epoch": epoch,
                "train_loss": float(sum(epoch_losses) / max(1, len(epoch_losses))),
                "train_acc": float(sum(epoch_accs) / max(1, len(epoch_accs))),
                "valid_loss": float(valid_loss),
                "valid_acc": float(valid_acc),
                "step": global_step,
                "lr": scheduler.get_last_lr()[0],
                "time": time.time(),
            }
            log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_f.flush()
            print(json.dumps(record, ensure_ascii=False))

            if valid_loss < (best_eval - args.early_stop_min_delta):
                best_eval = valid_loss
                best_epoch = epoch
                patience = 0
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                (best_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print(f"[stop] early stopping at epoch {epoch}")
                    break

    train_summary = {
        **run_config,
        "best_eval_loss": best_eval,
        "best_epoch": best_epoch,
        "total_optimizer_steps": global_step,
        "elapsed_sec": time.time() - t0,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
