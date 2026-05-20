# Ablation Study Guide

This directory contains the ablation workflow for CoT4POI. It is intentionally isolated from the main pipeline so that intermediate files, outputs, and analysis tables can be generated without changing the core training scripts.

## Directory Layout

```text
ablation_study/
├── configs/              # Dataset and experiment configs
├── cot_ablation/         # Macro and fine-stage CoT ablations
├── final_llm_ablation/   # Final LLM input-signal ablations
├── analysis/             # Metrics collection, tables, cases, and cost summaries
└── outputs/              # Generated outputs, ignored by Git
```

## Setup

Activate the project environment:

```bash
conda activate llm4poi
```

Preference generation uses API credentials from the project root `.env` file or shell environment variables.

Example `.env`:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
SILICONFLOW_API_KEY=...
SILICONFLOW_BASE_URL=...
```

`OPENAI_API_KEY` is enough for the main preference-generation script. The ablation preference module can also use SiliconFlow-compatible variables.

## Reusing Existing Full Results

The `full` variants can reuse existing main-pipeline outputs instead of regenerating them. The lookup order is:

1. paths configured in `ablation_study/configs/external_results.yml`
2. common paths under `datasets/<dataset>/...`

Check reuse status:

```bash
python ablation_study/analysis/external_results_status.py --dataset_name nyc
```

If auto-discovery fails, update:

```text
ablation_study/configs/external_results.yml
```

## CoT-Stage Ablations

### Macro Stage

```bash
python ablation_study/cot_ablation/run_macro_ablation.py --dataset_name nyc --variant full
python ablation_study/cot_ablation/run_macro_ablation.py --dataset_name nyc --variant w_o_td
python ablation_study/cot_ablation/run_macro_ablation.py --dataset_name nyc --variant w_o_preference
python ablation_study/cot_ablation/run_macro_ablation.py --dataset_name nyc --variant history_only
```

Replace `nyc` with `tky` or `ca` for other datasets.

Outputs:

```text
ablation_study/outputs/cot/<dataset>/macro/<variant>/
```

### Fine Stage

```bash
python ablation_study/cot_ablation/run_fine_ablation.py --dataset_name nyc --variant full
python ablation_study/cot_ablation/run_fine_ablation.py --dataset_name nyc --variant w_o_td
python ablation_study/cot_ablation/run_fine_ablation.py --dataset_name nyc --variant w_o_preference
python ablation_study/cot_ablation/run_fine_ablation.py --dataset_name nyc --variant w_o_macro
python ablation_study/cot_ablation/run_fine_ablation.py --dataset_name nyc --variant history_only
```

To reuse an existing full fine-stage model and only run evaluation:

```bash
python ablation_study/cot_ablation/run_fine_ablation.py --dataset_name nyc --variant full --skip_train
```

Outputs:

```text
ablation_study/outputs/cot/<dataset>/fine/<variant>/
```

## Final LLM Ablations

The final LLM ablation is usually run on `nyc` first because it is the fastest split for debugging.

### Build Ablation Inputs

```bash
python ablation_study/final_llm_ablation/build_inputs.py --dataset_name nyc --variant full
python ablation_study/final_llm_ablation/build_inputs.py --dataset_name nyc --variant w_o_fine
python ablation_study/final_llm_ablation/build_inputs.py --dataset_name nyc --variant w_o_macro
python ablation_study/final_llm_ablation/build_inputs.py --dataset_name nyc --variant w_o_preference
python ablation_study/final_llm_ablation/build_inputs.py --dataset_name nyc --variant w_o_td
```

Outputs:

```text
ablation_study/outputs/final_llm/nyc/<variant>/data/
```

### Train and Evaluate

Dry run:

```bash
python ablation_study/final_llm_ablation/run_end2end_ablation.py \
  --dataset_name nyc \
  --variant full \
  --skip_train \
  --skip_eval \
  --dry_run
```

Full run example:

```bash
python ablation_study/final_llm_ablation/run_end2end_ablation.py \
  --dataset_name nyc \
  --variant full \
  --model_name_or_path model/Llama-2-7b-longlora-32k-ft
```

Outputs:

```text
ablation_study/outputs/final_llm/nyc/<variant>/run/
```

## Input Masking Analysis

```bash
python ablation_study/final_llm_ablation/run_input_masking.py \
  --dataset_name nyc \
  --masked_field fine \
  --model_name_or_path model/Llama-2-7b-longlora-32k-ft \
  --lora_output_dir ablation_study/outputs/final_llm/nyc/full/run/checkpoint
```

Supported fields:

- `preference`
- `macro`
- `fine`

## Metrics and Tables

```bash
python ablation_study/analysis/collect_metrics.py --stage cot
python ablation_study/analysis/collect_metrics.py --stage final_llm
python ablation_study/analysis/make_tables.py --table_name cot_macro_table
python ablation_study/analysis/make_tables.py --table_name cot_fine_table
python ablation_study/analysis/make_tables.py --table_name final_llm_table
```

Table outputs are written to:

```text
ablation_study/outputs/tables/
```

## Case Studies and Cost Summary

```bash
python ablation_study/analysis/make_case_studies.py --dataset_name nyc
python ablation_study/analysis/cost_summary.py --dataset_name nyc --preference_hours 1.5 --fine_hours 5 --final_llm_hours 5
```

Outputs:

```text
ablation_study/outputs/cases/
ablation_study/outputs/tables/
```
