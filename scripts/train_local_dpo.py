#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_neuron.common import (  # noqa: E402
    group_neurons,
    load_model_and_tokenizer,
    load_top_neurons,
    read_jsonl,
)
from api_neuron.local_dpo import (  # noqa: E402
    attach_local_edits,
    apply_edit_payload,
    apply_state_dict_payload,
    export_ffn2_weight_payload,
    export_state_dict_payload,
    set_local_edits_enabled,
    trainable_parameters,
)


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


def retention_completion(row: dict) -> str:
    api = row.get("api", "")
    api_tail = api[len("torch.") :] if api.startswith("torch.") else api
    return (
        row.get("target_chain")
        or row.get("expected_completion")
        or row.get("base_predicted_chain")
        or api_tail
    )


def build_retention_batch(
    tokenizer,
    rows: Sequence[dict],
    max_length: int,
    max_prompt_length: int,
    max_completion_tokens: int,
):
    sequences, prompt_lengths, completion_lengths = [], [], []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids[-max_prompt_length:]
        max_completion = max(0, min(max_length - len(prompt_ids), max_completion_tokens))
        completion_ids = tokenizer(retention_completion(row), add_special_tokens=False).input_ids[:max_completion]
        sequences.append(prompt_ids + completion_ids)
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(len(completion_ids))

    pad_id = tokenizer.pad_token_id
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for idx, sequence in enumerate(sequences):
        input_ids[idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[idx, : len(sequence)] = 1
    return input_ids, attention_mask, torch.tensor(prompt_lengths), torch.tensor(completion_lengths)


def sequence_logps_and_counts(model, input_ids, attention_mask, prompt_lengths):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    labels = input_ids[:, 1:]
    token_logps = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    mask = (positions >= (prompt_lengths.unsqueeze(1) - 1)) & attention_mask[:, 1:].bool()
    counts = mask.sum(dim=1).clamp_min(1)
    return (token_logps * mask).sum(dim=1), counts


def sequence_logps(model, input_ids, attention_mask, prompt_lengths):
    logps, _ = sequence_logps_and_counts(model, input_ids, attention_mask, prompt_lengths)
    return logps


def retention_kl_loss(policy_logits, reference_logits, attention_mask, prompt_lengths, completion_lengths, kl_positions: int):
    seq_len = policy_logits.size(1)
    positions = torch.arange(seq_len, device=policy_logits.device).unsqueeze(0)
    start = prompt_lengths.unsqueeze(1) - 1
    if int(kl_positions) <= 0:
        span = completion_lengths.unsqueeze(1)
    else:
        span = torch.minimum(
            completion_lengths.unsqueeze(1),
            torch.full_like(completion_lengths.unsqueeze(1), int(kl_positions)),
        )
    end = start + torch.clamp(span, min=1)
    mask = (positions >= start) & (positions < end) & attention_mask.bool()
    if not mask.any():
        return policy_logits.new_zeros(())

    policy_selected = policy_logits[mask].float()
    reference_selected = reference_logits[mask].float()
    policy_log_probs = F.log_softmax(policy_selected, dim=-1)
    reference_probs = F.softmax(reference_selected, dim=-1)
    return F.kl_div(policy_log_probs, reference_probs, reduction="batchmean")


def retention_nll_loss(policy_logits, input_ids, attention_mask, prompt_lengths, completion_lengths, nll_positions: int):
    logits = policy_logits[:, :-1].float()
    labels = input_ids[:, 1:]
    label_mask = attention_mask[:, 1:].bool()
    positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    start = prompt_lengths.unsqueeze(1) - 1
    if int(nll_positions) <= 0:
        span = completion_lengths.unsqueeze(1)
    else:
        span = torch.minimum(
            completion_lengths.unsqueeze(1),
            torch.full_like(completion_lengths.unsqueeze(1), int(nll_positions)),
        )
    end = start + torch.clamp(span, min=1)
    mask = (positions >= start) & (positions < end) & label_mask
    if not mask.any():
        return logits.new_zeros(())

    token_nll = -torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_nll[mask].mean()


def dpo_loss(policy_logps, reference_logps, batch_size: int, beta: float):
    chosen_policy = policy_logps[:batch_size]
    rejected_policy = policy_logps[batch_size:]
    chosen_reference = reference_logps[:batch_size]
    rejected_reference = reference_logps[batch_size:]
    logits = beta * ((chosen_policy - rejected_policy) - (chosen_reference - rejected_reference))
    return -torch.nn.functional.logsigmoid(logits).mean()


def export_adapter_payload(model, model_path: str, neuron_file: Path | str, top_k: int, mode: str) -> dict:
    if mode == "down_only":
        return export_ffn2_weight_payload(
            model,
            model_path=model_path,
            neuron_file=neuron_file,
            top_k=top_k,
        )
    return export_state_dict_payload(
        model,
        model_path=model_path,
        neuron_file=neuron_file,
        top_k=top_k,
        mode=mode,
    )


def evaluate(
    model,
    loader,
    device,
    beta: float,
    retention_loader=None,
    retention_kl_weight: float = 0.0,
    kl_positions: int = 1,
    retention_nll_weight: float = 0.0,
    retention_nll_positions: int = 0,
):
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for input_ids, attention_mask, prompt_lengths, batch_size in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            prompt_lengths = prompt_lengths.to(device)
            set_local_edits_enabled(model, False)
            reference = sequence_logps(model, input_ids, attention_mask, prompt_lengths)
            set_local_edits_enabled(model, True)
            policy = sequence_logps(model, input_ids, attention_mask, prompt_lengths)
            loss = dpo_loss(policy, reference, batch_size=batch_size, beta=beta)
            total_loss += loss.item() * batch_size
            total_items += batch_size
        if retention_loader is not None and retention_kl_weight > 0:
            for input_ids, attention_mask, prompt_lengths, completion_lengths in retention_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                prompt_lengths = prompt_lengths.to(device)
                completion_lengths = completion_lengths.to(device)
                set_local_edits_enabled(model, False)
                reference_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                set_local_edits_enabled(model, True)
                policy_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                kl_loss = retention_kl_loss(
                    policy_logits,
                    reference_logits,
                    attention_mask,
                    prompt_lengths,
                    completion_lengths,
                    kl_positions=kl_positions,
                )
                loss = retention_kl_weight * kl_loss
                if retention_nll_weight > 0:
                    loss = loss + retention_nll_weight * retention_nll_loss(
                        policy_logits,
                        input_ids,
                        attention_mask,
                        prompt_lengths,
                        completion_lengths,
                        nll_positions=retention_nll_positions,
                    )
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_items += batch_size
    model.train()
    return total_loss / max(1, total_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neuron-local DPO adapter on a fixed set of FFN neurons.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--neuron-file", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--retention-train-file", type=Path, default=None)
    parser.add_argument("--retention-valid-file", type=Path, default=None)
    parser.add_argument("--init-adapter", type=Path, default=None, help="Optional existing neuron-local adapter to continue training from.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["down_only", "full"], default="full")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--retention-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--retention-kl-weight", type=float, default=0.0)
    parser.add_argument("--retention-kl-positions", type=int, default=1, help="Number of API completion token positions to constrain; 0 means the full tokenized API completion.")
    parser.add_argument("--retention-max-completion-tokens", type=int, default=8)
    parser.add_argument("--retention-nll-weight", type=float, default=0.0, help="Optional NLL anchor on retention API completions.")
    parser.add_argument("--retention-nll-positions", type=int, default=0, help="Number of retention completion token positions for NLL; 0 means all retained completion tokens.")
    parser.add_argument("--chosen-nll-weight", type=float, default=0.0, help="Optional token-average NLL anchor on chosen repair continuations.")
    parser.add_argument("--save-every-epoch", action="store_true", help="Also write an epoch checkpoint next to --output after each epoch.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=480)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_file)
    valid_rows = read_jsonl(args.valid_file)
    retention_train_rows = read_jsonl(args.retention_train_file) if args.retention_train_file else []
    retention_valid_rows = read_jsonl(args.retention_valid_file) if args.retention_valid_file else []
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, device=args.device, dtype=args.dtype)
    model.config.use_cache = False

    if args.init_adapter is not None:
        payload = torch.load(args.init_adapter, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            loaded_mode, missing, unexpected = apply_state_dict_payload(model, payload, mode_hint=args.mode)
            if loaded_mode != args.mode:
                print(f"[warn] init adapter mode={loaded_mode} but requested mode={args.mode}; continuing with loaded adapter.")
            if unexpected:
                print(f"[warn] unexpected init adapter keys: {len(unexpected)}")
            if missing:
                print(f"[info] missing base-model keys during init adapter load: {len(missing)}")
        elif isinstance(payload, dict) and "layers" in payload:
            apply_edit_payload(model, payload)
        else:
            raise ValueError(f"Unsupported init adapter payload format: {args.init_adapter}")
    else:
        neurons = load_top_neurons(args.neuron_file, top_k=args.top_k)
        attach_local_edits(model, group_neurons(neurons), mode=args.mode)
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)

    train_loader = DataLoader(
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda rows: build_batch(tokenizer, rows, args.max_length, args.max_prompt_length),
    )
    valid_loader = DataLoader(
        valid_rows,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda rows: build_batch(tokenizer, rows, args.max_length, args.max_prompt_length),
    )
    retention_train_loader = None
    if retention_train_rows and args.retention_kl_weight > 0:
        retention_train_loader = DataLoader(
            retention_train_rows,
            batch_size=args.retention_batch_size or args.batch_size,
            shuffle=True,
            collate_fn=lambda rows: build_retention_batch(
                tokenizer,
                rows,
                args.max_length,
                args.max_prompt_length,
                args.retention_max_completion_tokens,
            ),
        )
    retention_valid_loader = None
    if retention_valid_rows and args.retention_kl_weight > 0:
        retention_valid_loader = DataLoader(
            retention_valid_rows,
            batch_size=args.retention_batch_size or args.batch_size,
            shuffle=False,
            collate_fn=lambda rows: build_retention_batch(
                tokenizer,
                rows,
                args.max_length,
                args.max_prompt_length,
                args.retention_max_completion_tokens,
            ),
        )

    best_loss = float("inf")
    history = []
    step = 0
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        updates_since_zero = 0
        retention_iter = iter(retention_train_loader) if retention_train_loader is not None else None
        for batch_idx, (input_ids, attention_mask, prompt_lengths, batch_size) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            prompt_lengths = prompt_lengths.to(device)

            with torch.no_grad():
                set_local_edits_enabled(model, False)
                reference = sequence_logps(model, input_ids, attention_mask, prompt_lengths)

            set_local_edits_enabled(model, True)
            policy, policy_counts = sequence_logps_and_counts(model, input_ids, attention_mask, prompt_lengths)
            loss = dpo_loss(policy, reference, batch_size=batch_size, beta=args.beta)
            dpo_value = loss.detach().float().item()
            retention_value = 0.0
            retention_nll_value = 0.0
            chosen_nll_value = 0.0
            if args.chosen_nll_weight > 0:
                chosen_nll = -(policy[:batch_size] / policy_counts[:batch_size].float()).mean()
                chosen_nll_value = chosen_nll.detach().float().item()
                loss = loss + args.chosen_nll_weight * chosen_nll
            if retention_iter is not None:
                try:
                    ret_input_ids, ret_attention_mask, ret_prompt_lengths, ret_completion_lengths = next(retention_iter)
                except StopIteration:
                    retention_iter = iter(retention_train_loader)
                    ret_input_ids, ret_attention_mask, ret_prompt_lengths, ret_completion_lengths = next(retention_iter)
                ret_input_ids = ret_input_ids.to(device)
                ret_attention_mask = ret_attention_mask.to(device)
                ret_prompt_lengths = ret_prompt_lengths.to(device)
                ret_completion_lengths = ret_completion_lengths.to(device)
                with torch.no_grad():
                    set_local_edits_enabled(model, False)
                    ret_reference_logits = model(input_ids=ret_input_ids, attention_mask=ret_attention_mask).logits
                set_local_edits_enabled(model, True)
                ret_policy_logits = model(input_ids=ret_input_ids, attention_mask=ret_attention_mask).logits
                ret_loss = retention_kl_loss(
                    ret_policy_logits,
                    ret_reference_logits,
                    ret_attention_mask,
                    ret_prompt_lengths,
                    ret_completion_lengths,
                    kl_positions=args.retention_kl_positions,
                )
                retention_value = ret_loss.detach().float().item()
                loss = loss + args.retention_kl_weight * ret_loss
                if args.retention_nll_weight > 0:
                    ret_nll = retention_nll_loss(
                        ret_policy_logits,
                        ret_input_ids,
                        ret_attention_mask,
                        ret_prompt_lengths,
                        ret_completion_lengths,
                        nll_positions=args.retention_nll_positions,
                    )
                    retention_nll_value = ret_nll.detach().float().item()
                    loss = loss + args.retention_nll_weight * ret_nll
            (loss / args.grad_accum).backward()
            updates_since_zero += 1

            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                updates_since_zero = 0
                if step % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch + 1,
                                "dpo_loss": dpo_value,
                                "retention_kl": retention_value,
                                "retention_nll": retention_nll_value,
                                "chosen_nll": chosen_nll_value,
                                "loss": loss.detach().float().item(),
                            }
                        ),
                        flush=True,
                    )

        if updates_since_zero:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

        valid_loss = evaluate(
            model,
            valid_loader,
            device=device,
            beta=args.beta,
            retention_loader=retention_valid_loader,
            retention_kl_weight=args.retention_kl_weight,
            kl_positions=args.retention_kl_positions,
            retention_nll_weight=args.retention_nll_weight,
            retention_nll_positions=args.retention_nll_positions,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "valid_loss": valid_loss,
                "retention_kl_weight": args.retention_kl_weight,
                "retention_kl_positions": args.retention_kl_positions,
                "retention_nll_weight": args.retention_nll_weight,
                "retention_nll_positions": args.retention_nll_positions,
                "retention_train_n": len(retention_train_rows),
                "retention_valid_n": len(retention_valid_rows),
                "chosen_nll_weight": args.chosen_nll_weight,
            }
        )
        print(json.dumps(history[-1]))
        if args.save_every_epoch:
            epoch_output = args.output.with_name(f"{args.output.stem}.epoch{epoch + 1}{args.output.suffix}")
            epoch_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                export_adapter_payload(
                    model,
                    model_path=args.model_path,
                    neuron_file=args.neuron_file,
                    top_k=args.top_k,
                    mode=args.mode,
                ),
                epoch_output,
            )
        if valid_loss < best_loss:
            best_loss = valid_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            payload = export_adapter_payload(
                    model,
                    model_path=args.model_path,
                    neuron_file=args.neuron_file,
                    top_k=args.top_k,
                    mode=args.mode,
            )
            torch.save(payload, args.output)

    log_path = args.output.with_suffix(".log.json")
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
