from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import get_scheduler

from .common import get_layers


def select_top_layers(
    layer_counts_file: Path,
    model_preset: str,
    seed: str = "matrix",
    top_n: int = 3,
) -> List[int]:
    rows: List[Tuple[int, int]] = []
    with layer_counts_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["model"] != model_preset or row["seed"] != seed:
                continue
            rows.append((int(row["layer"]), int(row["count"])))
    if not rows:
        raise ValueError(
            f"No layer counts found for model_preset={model_preset!r}, seed={seed!r} "
            f"in {layer_counts_file}"
        )
    rows.sort(key=lambda item: (-item[1], -item[0]))
    return [layer for layer, _ in rows[:top_n]]


def infer_lora_targets(model) -> tuple[list[str], str]:
    layers = get_layers(model)
    if not layers:
        raise ValueError("Could not find transformer layers for LoRA target selection.")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_pattern = "layers"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers_pattern = "h"
    else:
        raise ValueError("Unsupported transformer layout for LoRA layer pattern selection.")
    mlp = layers[0].mlp
    if hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
        return ["c_fc", "c_proj"], layers_pattern
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        return ["gate_proj", "up_proj", "down_proj"], layers_pattern
    raise ValueError(f"Unsupported MLP type for LoRA targeting: {mlp.__class__.__name__}")


def attach_layer_lora(
    model,
    layers_to_transform: Sequence[int],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    target_modules, layers_pattern = infer_lora_targets(model)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        layers_to_transform=list(layers_to_transform),
        layers_pattern=layers_pattern,
    )
    model = get_peft_model(model, config)
    return model, config


def build_sft_batch(
    tokenizer,
    rows: Sequence[dict],
    max_length: int,
    max_prompt_length: int,
    add_eos: bool = True,
):
    encoded_rows = []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer pad_token_id must be set before batching.")

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
        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        label_tensor = torch.tensor(label_ids, dtype=torch.long)
        input_ids[idx, : len(sequence)] = seq_tensor
        attention_mask[idx, : len(sequence)] = 1
        labels[idx, : len(label_ids)] = label_tensor
    return input_ids, attention_mask, labels


def completion_loss(model, input_ids, attention_mask, labels):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    target = labels[:, 1:]
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=-100,
    )


def evaluate_loss(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            loss = completion_loss(model, input_ids, attention_mask, labels)
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
    model.train()
    return total_loss / max(1, total_items)


def count_trainable_parameters(model) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return trainable, total


def build_scheduler(optimizer, scheduler_type: str, warmup_ratio: float, total_steps: int):
    warmup_steps = int(math.ceil(total_steps * warmup_ratio))
    return get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def maybe_enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False


def rows_to_sft_preview(rows: Iterable[dict], limit: int = 3) -> list[dict]:
    preview = []
    for idx, row in enumerate(rows):
        if idx >= limit:
            break
        preview.append(
            {
                "prompt_tail": row["prompt"][-120:],
                "chosen": row["chosen"],
                "rejected": row.get("rejected"),
            }
        )
    return preview
