#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from api_neuron.common import (  # noqa: E402
    build_seed_direction,
    clean_token,
    decode_direction,
    get_ffn2_weight,
    get_layers,
    get_unembedding,
    is_readable_token,
    load_model_and_tokenizer,
)


OVERLAP_TARGETS = {
    "starcoder2-15b": {
        "model_path": "/data/lkl/models/StarCoder/starcoder2-15b",
        "neurons": [(39, 24502)],
    },
    "deepseek-coder-6.7b-instruct": {
        "model_path": "/data/lkl/models/deepseek-ai/deepseek-coder-6.7b-instruct",
        "neurons": [
            (30, 1178),
            (30, 2358),
            (30, 4238),
            (30, 4268),
            (30, 9244),
            (31, 1305),
            (31, 1524),
            (31, 2347),
            (31, 5309),
            (31, 6245),
            (31, 8569),
        ],
    },
}

SEED_TERMS = [
    "torch",
    "linalg",
    "norm",
    "svd",
    "cholesky",
    "symeig",
    "eigh",
    "matrix",
]

TORCH_KEYWORDS = [
    "torch",
    "linalg",
    "norm",
    "svd",
    "cholesky",
    "symeig",
    "eigh",
    "matrix",
    "tensor",
    "eig",
    "diag",
]


def run_localize(args: argparse.Namespace) -> None:
    model, tokenizer, _ = load_model_and_tokenizer(args.model_path, device=args.device, dtype=args.dtype)
    seed_direction = build_seed_direction(tokenizer, model, args.seed)

    scored = []
    for layer_idx, layer in enumerate(get_layers(model)):
        weight = get_ffn2_weight(layer).float()
        scores = torch.matmul(weight, seed_direction)
        values, indices = torch.topk(scores, k=min(scores.numel(), args.top_k))
        for score, neuron_idx in zip(values.tolist(), indices.tolist()):
            direction = weight[neuron_idx]
            scored.append(
                {
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "score": float(score),
                    "decoded_tokens": decode_direction(tokenizer, model, direction, top_n=args.decode_top),
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    top_neurons = scored[: args.top_k]
    layer_counts = Counter(item["layer"] for item in top_neurons)

    payload = {
        "model_path": args.model_path,
        "seed_strings": args.seed,
        "top_k": args.top_k,
        "global_top_neurons": top_neurons,
        "layer_counts": dict(sorted(layer_counts.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["layer_counts"], indent=2))


def decode_direction_with_scores(tokenizer, model, direction: torch.Tensor, top_n: int) -> List[dict]:
    unembedding = get_unembedding(model).float()
    scores = torch.matmul(unembedding, direction.float())
    values, indices = torch.topk(scores, k=min(scores.numel(), top_n * 8))
    out: List[dict] = []
    seen = set()
    for score, token_id in zip(values.tolist(), indices.tolist()):
        token_raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        token = clean_token(token_raw)
        if not is_readable_token(token) or token in seen:
            continue
        seen.add(token)
        out.append({"token": token, "token_id": int(token_id), "score": float(score)})
        if len(out) >= top_n:
            break
    return out


def is_torch_related(token: str) -> bool:
    low = token.lower()
    return any(keyword in low for keyword in TORCH_KEYWORDS)


def compute_seed_projections(direction: torch.Tensor, tokenizer, model, terms: Sequence[str]) -> Dict[str, float]:
    projections: Dict[str, float] = {}
    for term in terms:
        seed_vec = build_seed_direction(tokenizer, model, [term]).float()
        projections[term] = float(torch.dot(direction.float(), seed_vec).item())
    return projections


def run_decode_overlap(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []

    for model_name, spec in OVERLAP_TARGETS.items():
        model_path = str(spec["model_path"])
        neurons: Sequence[Tuple[int, int]] = list(spec["neurons"])
        print(f"[decode] loading {model_name} from {model_path}")
        model, tokenizer, _ = load_model_and_tokenizer(model_path, device=args.device, dtype=args.dtype)
        layers = get_layers(model)

        model_rows: List[dict] = []
        for layer_idx, neuron_idx in neurons:
            direction = get_ffn2_weight(layers[layer_idx])[int(neuron_idx)].detach().float()
            top_tokens = decode_direction_with_scores(tokenizer, model, direction, top_n=args.top_n)
            torch_tokens = [item for item in top_tokens if is_torch_related(item["token"])]
            projections = compute_seed_projections(direction, tokenizer, model, SEED_TERMS)
            row = {
                "model": model_name,
                "model_path": model_path,
                "layer": int(layer_idx),
                "neuron": int(neuron_idx),
                "top_tokens": top_tokens,
                "torch_related_tokens": torch_tokens,
                "seed_projections": projections,
            }
            model_rows.append(row)
            all_rows.append(row)
            print(f"  - ({layer_idx}, {neuron_idx}) torch_related_top{args.top_n}={len(torch_tokens)}")

        model_out = args.output_dir / f"{model_name}.decoded.json"
        model_out.write_text(json.dumps(model_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {model_out}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_out = args.output_dir / "all_overlap_neurons.decoded.json"
    all_out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Overlap Neuron Decoding (Torch-related)",
        "",
        f"- Total neurons decoded: {len(all_rows)}",
        "",
        f"| Model | Layer | Neuron | Torch-related token count (top{args.top_n}) | Top torch-related tokens (up to 12) |",
        "|---|---:|---:|---:|---|",
    ]
    for row in all_rows:
        toks = [item["token"] for item in row["torch_related_tokens"][:12]]
        md_lines.append(
            f"| {row['model']} | {row['layer']} | {row['neuron']} | "
            f"{len(row['torch_related_tokens'])} | {', '.join(toks)} |"
        )
    md_out = args.output_dir / "summary.md"
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[saved] {all_out}")
    print(f"[saved] {md_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ1 API-specific neuron localization and decoding utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    localize = subparsers.add_parser("localize", help="Localize top-aligned API neurons from FFN2 write-back vectors.")
    localize.add_argument("--model-path", required=True, help="HF model path or model id.")
    localize.add_argument("--seed", action="append", required=True, help="Seed string. Pass multiple times for multiple seeds.")
    localize.add_argument("--top-k", type=int, default=200, help="Number of neurons to keep.")
    localize.add_argument("--decode-top", type=int, default=8, help="Number of decoded tokens to store for each top neuron.")
    localize.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    localize.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    localize.add_argument("--output", type=Path, required=True)
    localize.set_defaults(func=run_localize)

    decode = subparsers.add_parser("decode-overlap", help="Decode representative matrix/linalg overlap neurons.")
    decode.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "overlap_neuron_decode_20260512")
    decode.add_argument("--top-n", type=int, default=500)
    decode.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    decode.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    decode.set_defaults(func=run_decode_overlap)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
