#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "starcoder2-3b",
    "starcoder2-7b",
    "starcoder2-15b",
    "deepseek-coder-6.7b-instruct",
    "codellama-7b",
}

PAPER_TABLES = {
    "table_i_datasets.csv",
    "table_ii_matrix_linalg_overlap.csv",
    "table_iii_decode_examples.csv",
    "table_iv_targeted_ablation.csv",
    "table_v_k_sensitivity_matrix_set.csv",
    "table_vi_ablation_distribution_example.csv",
    "table_vii_specificity_set_ablation.csv",
    "table_viii_rq3_method_comparison.csv",
    "table_ix_migration_set_generalization.csv",
}

MIGRATION_PAIRS = {
    "torch_norm_to_linalg_norm",
    "torch_svd_to_linalg_svd",
    "torch_cholesky_to_linalg_cholesky",
    "torch_symeig_to_linalg_eigh",
}

MIGRATION_TASKS = {"norm", "svd", "cholesky", "symeig"}

FORBIDDEN_TEXT = {
    "/home/" + "fdse",
    "api-" + "neuron-repo",
    "torch_chain_matmul" + "_to_linalg_multi_dot",
    "scipy_interp2d" + "_to_bisplev",
    "sklearn_gmm" + "_to_gaussian_mixture",
}


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def require_file(errors: list[str], rel: str) -> Path:
    path = REPO_ROOT / rel
    if not path.is_file():
        fail(errors, f"missing file: {rel}")
    return path


def require_dir(errors: list[str], rel: str) -> Path:
    path = REPO_ROOT / rel
    if not path.is_dir():
        fail(errors, f"missing directory: {rel}")
    return path


def check_no_cache_or_old_dirs(errors: list[str]) -> None:
    blocked_dirs = [
        "checkpoints",
        "data/external_pairs",
        "results/dpo",
        "outputs",
    ]
    for rel in blocked_dirs:
        if (REPO_ROOT / rel).exists():
            fail(errors, f"unexpected release directory: {rel}")

    for path in REPO_ROOT.rglob("*"):
        if path.name == "__pycache__":
            fail(errors, f"unexpected __pycache__: {path.relative_to(REPO_ROOT)}")
        if path.suffix == ".pyc":
            fail(errors, f"unexpected .pyc file: {path.relative_to(REPO_ROOT)}")


def check_text_hygiene(errors: list[str]) -> None:
    suffixes = {".csv", ".json", ".md", ".py", ".txt"}
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in FORBIDDEN_TEXT:
            if marker in text:
                fail(errors, f"forbidden text {marker!r} in {path.relative_to(REPO_ROOT)}")


def check_main_case(errors: list[str]) -> None:
    matrix = require_file(errors, "data/main_case/matrix_set.jsonl")
    linalg = require_file(errors, "data/main_case/linalg_set.jsonl")
    require_file(errors, "data/main_case/matrix_namespace_prompts.jsonl")
    require_file(errors, "data/main_case/linalg_namespace_prompts.jsonl")
    if matrix.exists() and count_jsonl(matrix) != 1000:
        fail(errors, "Matrix Set should contain 1000 prompts")
    if linalg.exists() and count_jsonl(linalg) != 5291:
        fail(errors, "Linalg Set should contain 5291 prompts")

    for model in MODELS:
        for split in ("train", "valid", "test"):
            require_file(errors, f"data/main_case/dpo/{model}/{split}.jsonl")


def check_migration_set(errors: list[str]) -> None:
    raw_root = require_dir(errors, "data/migration_set/raw_1000_per_pair")
    for pair in MIGRATION_PAIRS:
        path = raw_root / pair / "all.jsonl"
        if not path.is_file():
            fail(errors, f"missing Migration Set all.jsonl for {pair}")
        elif count_jsonl(path) != 1000:
            fail(errors, f"Migration Set pair {pair} should contain 1000 prompts")

    norm_svd_root = require_dir(errors, "data/migration_set/model_conditioned_norm_svd")
    chol_sym_root = require_dir(errors, "data/migration_set/model_conditioned_cholesky_symeig")
    for model in MODELS:
        for task in ("norm", "svd"):
            for split in ("train", "valid", "test"):
                require_file(errors, f"{norm_svd_root.relative_to(REPO_ROOT)}/{model}/{task}/{split}.jsonl")
        for task in ("cholesky", "symeig"):
            for split in ("train", "valid", "test"):
                require_file(errors, f"{chol_sym_root.relative_to(REPO_ROOT)}/{model}/{task}/{split}.jsonl")


def check_specificity_set(errors: list[str]) -> None:
    root = require_dir(errors, "data/specificity_set/migration_pair_80api_2400")
    for task in MIGRATION_TASKS:
        path = root / f"{task}_task" / "all.jsonl"
        if not path.is_file():
            fail(errors, f"missing Specificity Set all.jsonl for {task}")
        elif count_jsonl(path) != 2400:
            fail(errors, f"Specificity Set {task}_task should contain 2400 prompts")

    filtered_root = require_dir(errors, "data/specificity_set/main_case_model_filtered_80api")
    for model in MODELS:
        for split in ("train", "valid", "test"):
            require_file(errors, f"{filtered_root.relative_to(REPO_ROOT)}/{model}/{split}.jsonl")


def check_results(errors: list[str]) -> None:
    table_root = require_dir(errors, "results/paper_tables")
    existing = {path.name for path in table_root.glob("*.csv")}
    missing = PAPER_TABLES - existing
    extra = existing - PAPER_TABLES
    if missing:
        fail(errors, f"missing paper table CSVs: {sorted(missing)}")
    if extra:
        fail(errors, f"unexpected paper table CSVs: {sorted(extra)}")

    for table in PAPER_TABLES:
        path = table_root / table
        if path.exists() and not read_csv(path):
            fail(errors, f"empty paper table: {table}")

    for seed in ("matrix", "linalg"):
        for model in MODELS:
            require_file(errors, f"results/localization/{seed}/{model}.top200.json")
    require_file(errors, "results/localization/top200_layer_counts.csv")

    for rel in (
        "results/rq1/table_ii_matrix_linalg_overlap.csv",
        "results/rq2/table_iv_targeted_ablation.csv",
        "results/rq2/table_v_k_sensitivity_matrix_set.csv",
        "results/rq3/table_viii_rq3_method_comparison.csv",
        "results/rq3/table_ix_migration_set_generalization.csv",
    ):
        require_file(errors, rel)


def main() -> None:
    errors: list[str] = []
    check_no_cache_or_old_dirs(errors)
    check_main_case(errors)
    check_migration_set(errors)
    check_specificity_set(errors)
    check_results(errors)
    check_text_hygiene(errors)

    if errors:
        print("Release audit failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Release audit passed.")


if __name__ == "__main__":
    main()
