# api-neuron-loc-edit

## Repository Overview

| Path | Description |
|------|-------------|
| `data/` | Processed datasets used by the paper. Dataset construction scripts are intentionally omitted. |
| `results/localization/` | Localized top-200 neuron sets for the `matrix` and `linalg` seeds. |
| `results/paper_tables/` | CSV copies of the paper tables. |
| `results/rq1/` | RQ1 layer distribution, overlap, and decoding summaries. |
| `results/rq2/` | RQ2 ablation and K-sensitivity summaries. |
| `results/rq3/` | RQ3 editing, specificity, and migration summaries. |
| `scripts/` | Minimal RQ1/RQ2/RQ3 experiment entry points. |
| `src/api_neuron/` | Shared implementation modules. |

## Dataset Preparation

The processed datasets used in the paper are stored under `data/`. No external dataset download is required for the released artifact.

| Dataset | File or Directory | Size | Used For |
|---------|-------------------|------|----------|
| Matrix Set | `data/main_case/matrix_set.jsonl` | 1,000 prompts | RQ1 localization, RQ2 causal validation, RQ3 primary mitigation case |
| Linalg Set | `data/main_case/linalg_set.jsonl` | 5,291 prompts | RQ1 localization and RQ2 replacing-API causal validation |
| Namespace prompts | `data/main_case/matrix_namespace_prompts.jsonl`, `data/main_case/linalg_namespace_prompts.jsonl` | Processed prompts | Localization, ablation, and editing evaluation |
| Main DPO splits | `data/main_case/dpo/<model>/{train,valid,test}.jsonl` | Model-specific splits | Primary matrix-to-linalg DPO training and evaluation |
| Main Specificity Set | `data/specificity_set/main_case_model_filtered_80api/<model>/` | Model-specific splits | Specificity-KL and specificity evaluation |
| Migration Specificity Set | `data/specificity_set/migration_pair_80api_2400/` | 2,400 prompts per task | Specificity control for migration experiments |
| Migration Set | `data/migration_set/raw_1000_per_pair/` | 4,000 prompts | Generalization to additional deprecated-replacing API pairs |
| Model-conditioned migration splits | `data/migration_set/model_conditioned_norm_svd/`, `data/migration_set/model_conditioned_cholesky_symeig/` | Model-specific splits | Table IX experiments |

The Migration Set covers the following PyTorch deprecated-replacing API pairs:

| Deprecated API | Replacing API |
|----------------|---------------|
| `torch.norm` | `torch.linalg.norm` |
| `torch.svd` | `torch.linalg.svd` |
| `torch.cholesky` | `torch.linalg.cholesky` |
| `torch.symeig` | `torch.linalg.eigh` |

## Environment Setup

All commands below assume they are run from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The base model weights are not included in this repository. The scripts expect HuggingFace-compatible model paths.

| Model Alias | HuggingFace Model |
|-------------|-------------------|
| StarCoder2-3B | `bigcode/starcoder2-3b` |
| StarCoder2-7B | `bigcode/starcoder2-7b` |
| StarCoder2-15B | `bigcode/starcoder2-15b` |
| DeepSeek-Coder | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| CodeLlama | `codellama/CodeLlama-7b-hf` |

## Reproducing RQ1: API-Specific Neuron Localization

RQ1 localizes API-specific neurons and analyzes their layer distribution, overlap, and vocabulary-space projections.

### 1. Localize API-Specific Neurons

Run:

```bash
python3 scripts/rq1_neurons.py localize \
  --model-path bigcode/starcoder2-3b \
  --seed matrix \
  --top-k 200 \
  --output outputs/starcoder2-3b.matrix.top200.json
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-path` | HuggingFace model id or local model path. |
| `--seed` | Seed API string, such as `matrix` or `linalg`. Can be passed multiple times. |
| `--top-k` | Number of globally top-ranked neurons to keep. |
| `--decode-top` | Number of decoded vocabulary tokens stored for each localized neuron. |
| `--output` | Output JSON file containing localized neurons and layer counts. |

### 2. Decode Representative Overlap Neurons

Run:

```bash
python3 scripts/rq1_neurons.py decode-overlap \
  --output-dir outputs/overlap_neuron_decode
```

### RQ1 Scripts

| Script | Subcommand | Purpose |
|--------|------------|---------|
| `scripts/rq1_neurons.py` | `localize` | Localizes top-ranked API-specific FFN2 neurons for a seed API string. |
| `scripts/rq1_neurons.py` | `decode-overlap` | Decodes representative matrix/linalg overlap neurons into vocabulary-space tokens. |

## Reproducing RQ2: Causal Effect of Localized Neurons

RQ2 evaluates whether localized neurons causally affect API completion by comparing localized-neuron interventions with size-matched random-neuron interventions.

### 1. Targeted Ablation on Matrix or Linalg Prompts

Run:

```bash
python3 scripts/run_targeted_ablation.py \
  --model-path bigcode/starcoder2-3b \
  --neuron-file results/localization/matrix/starcoder2-3b.top200.json \
  --dataset data/main_case/matrix_namespace_prompts.jsonl \
  --mode gaussian \
  --output outputs/starcoder2-3b.matrix.gaussian.json
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-path` | HuggingFace model id or local model path. |
| `--neuron-file` | Localized neuron JSON file. |
| `--dataset` | Namespace-completion prompt file. |
| `--mode` | Intervention type: `zero` or `gaussian`. |
| `--top-k` | Number of localized neurons to intervene on. |
| `--random-runs` | Number of size-matched random neuron sets for comparison. |
| `--output` | Output JSON file for ablation results. |

### 2. K-Sensitivity Analysis

Run:

```bash
python3 scripts/run_matrix_prompt_k_ablation.py \
  --model-key starcoder2-3b \
  --top-k 100 \
  --output outputs/starcoder2-3b.matrix.k100.json
```

### 3. Non-Target API Retention Under Ablation

Run:

```bash
python3 scripts/evaluate_other_torch_retention_ablation.py \
  --model starcoder2-3b \
  --output-dir outputs/starcoder2-3b.retention_ablation
```

### RQ2 Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_targeted_ablation.py` | Runs zero-out or Gaussian intervention on localized neurons and random baselines. |
| `scripts/run_matrix_prompt_k_ablation.py` | Repeats Matrix Set ablation for alternative top-K values. |
| `scripts/evaluate_other_torch_retention_ablation.py` | Measures effects of matrix-localized neuron ablation on non-target PyTorch API completions. |

## Reproducing RQ3: Deprecated API Mitigation and Specificity

RQ3 trains and evaluates neuron-level editors for deprecated API mitigation while preserving non-target API completions.

### 1. Train the Primary Neuron-Level DPO Editor

Run:

```bash
python3 scripts/train_local_dpo.py \
  --model-path bigcode/starcoder2-3b \
  --neuron-file results/localization/matrix/starcoder2-3b.top200.json \
  --train-file data/main_case/dpo/starcoder2-3b/train.jsonl \
  --valid-file data/main_case/dpo/starcoder2-3b/valid.jsonl \
  --mode full \
  --output outputs/starcoder2-3b.full.pt
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-path` | HuggingFace model id or local model path. |
| `--neuron-file` | Localized neuron JSON file. |
| `--train-file` | DPO training split. |
| `--valid-file` | DPO validation split. |
| `--retention-train-file` | Optional Specificity-KL training file. |
| `--retention-valid-file` | Optional Specificity-KL validation file. |
| `--mode` | Local edit parameterization: `down_only` or `full`. |
| `--retention-kl-weight` | Specificity-KL weight. |
| `--output` | Saved local-edit payload. |

### 2. Train Baselines

| Method | Script | Purpose |
|--------|--------|---------|
| Neuron-SFT | `scripts/train_local_sft.py` | Trains a neuron-local supervised baseline using replacing API completions. |
| LoRA-DPO | `scripts/train_layer_lora_dpo.py` | Trains a layer-localized LoRA-DPO baseline on neuron-dense layers. |

### 3. Evaluate API Completion and Specificity

| Evaluation | Script | Typical Input |
|------------|--------|---------------|
| Local DPO chosen/rejected completion | `scripts/evaluate_api_completion.py local-dpo` | Saved `.pt` local-edit payload |
| LoRA-DPO chosen/rejected completion | `scripts/evaluate_api_completion.py lora-dpo` | LoRA adapter directory |
| Local DPO non-target API retention | `scripts/evaluate_local_other_api_retention.py` | Saved `.pt` local-edit payload |
| Multi-variant non-target API completion | `scripts/evaluate_other_api_completion.py` | Model preset and Specificity Set split |
| Local edit HumanEval/HumanEval+ | `scripts/evaluate_code_generation.py local-edit` | Saved `.pt` local-edit payload |
| LoRA HumanEval/HumanEval+ | `scripts/evaluate_code_generation.py lora` | LoRA adapter directory |

Example:

```bash
python3 scripts/evaluate_api_completion.py local-dpo \
  --adapter outputs/starcoder2-3b.full.pt \
  --data-file data/main_case/dpo/starcoder2-3b/test.jsonl \
  --output outputs/starcoder2-3b.full.test.jsonl \
  --compare-base
```

### RQ3 Usage Order

| Step | Action | Main Script |
|------|--------|-------------|
| 1 | Train Neuron-DPO editor | `train_local_dpo.py` |
| 2 | Train Neuron-SFT and LoRA-DPO baselines | `train_local_sft.py`, `train_layer_lora_dpo.py` |
| 3 | Evaluate deprecated/replacing API completion | `evaluate_api_completion.py` |
| 4 | Evaluate non-target API specificity | `evaluate_local_other_api_retention.py`, `evaluate_other_api_completion.py` |
| 5 | Evaluate general code generation ability | `evaluate_code_generation.py` |

## Paper Tables and Stored Results

CSV versions of the paper tables are stored in `results/paper_tables/`.

| Paper Table | File |
|-------------|------|
| Table I | `table_i_datasets.csv` |
| Table II | `table_ii_matrix_linalg_overlap.csv` |
| Table III | `table_iii_decode_examples.csv` |
| Table IV | `table_iv_targeted_ablation.csv` |
| Table V | `table_v_k_sensitivity_matrix_set.csv` |
| Table VI | `table_vi_ablation_distribution_example.csv` |
| Table VII | `table_vii_specificity_set_ablation.csv` |
| Table VIII | `table_viii_rq3_method_comparison.csv` |
| Table IX | `table_ix_migration_set_generalization.csv` |

## Release Audit

Run:

```bash
python3 scripts/audit_release_repo.py
```

The audit checks that the release contains the expected processed datasets and result tables, excludes cache/checkpoint/output directories, and does not contain known stale experimental artifacts.

## Notes

| Item | Policy |
|------|--------|
| Base model weights | Not included. Use HuggingFace model ids or local model paths. |
| Trained edit checkpoints | Not included. Rerun training scripts to regenerate them under `outputs/`. |
| `outputs/` | Not shipped. Scripts create this directory locally when rerun. |
| Dataset construction scripts | Not included. The release keeps processed datasets for reproduction. |
| Old cross-library experiments | Not included because they are outside the current paper scope. |
