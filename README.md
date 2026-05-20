# CoT4POI

CoT4POI is a research prototype for **Next Point-of-Interest (Next-POI) recommendation**. It explores how large language models can be used beyond direct ID prediction by adding explicit intermediate reasoning signals: temporal semantics, preference abstraction, macro-category prediction, and fine-grained category reranking.

The project was developed for an undergraduate thesis on spatio-temporal commonsense reasoning and hierarchical chain-of-thought recommendation.

## Why This Project

Most Next-POI recommenders represent places as IDs or dense embeddings. This works for pattern fitting, but it makes user intent hard to inspect and can struggle with sparse, long-tail POIs. CoT4POI adds a lightweight reasoning pipeline before the final POI prediction:

```text
trajectory -> temporal semantics -> preference -> macro category -> fine category -> POI ID
```

The goal is not to make the LLM "think aloud" for users, but to turn implicit mobility patterns into structured signals that can be tested, ablated, and injected into the final recommendation model.

## Main Ideas

- **Temporal semantic discretization**: converts raw timestamps into human-readable temporal context such as time-of-day and day-type signals.
- **Preference abstraction**: uses an LLM to summarize recent trajectory behavior into a concise user preference description.
- **Macro-to-fine reasoning**: first predicts coarse behavioral categories, then reranks fine-grained POI categories.
- **Final LLM recommendation**: injects preference, macro, and fine signals into the final supervised fine-tuning data for POI ID prediction.
- **Ablation workspace**: keeps module-level ablations separate from the main pipeline.

## Repository Structure

```text
CoT4POI/
├── preprocessing/                 # Raw check-in preprocessing
├── cot/                           # Main CoT-style preference/macro/fine pipeline
├── ablation_study/                # Ablation experiments and analysis scripts
├── ds_configs/                    # DeepSpeed configs
├── merge_for_sft.py               # Build final SFT data from CoT outputs
├── supervised-fine-tune-qlora.py  # QLoRA fine-tuning script
├── eval_next_poi.py               # Final Next-POI evaluation script
├── environment.yml                # Conda environment
└── README.md
```

Large local assets are intentionally not committed:

- raw and processed datasets
- model checkpoints
- API keys and `.env` files
- experiment outputs and logs

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate llm4poi
```

Preference generation requires an API key. Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then fill in the key you use:

```bash
OPENAI_API_KEY=...
```

The ablation code also supports SiliconFlow-compatible variables:

```bash
SILICONFLOW_API_KEY=...
SILICONFLOW_BASE_URL=...
```

## Data and Models

The experiments use three datasets:

- `nyc`
- `tky`
- `ca`

Place raw data under:

```text
datasets/nyc/raw/
datasets/tky/raw/
datasets/ca/raw/
```

The final LLM stage was designed around a long-context Llama-2 checkpoint, for example:

- `Yukang/Llama-2-7b-longlora-32k-ft`

Store local models under `model/`, or pass the actual path through command-line arguments.

## Main Pipeline

Replace `{dataset_name}` with `nyc`, `tky`, or `ca`.

### 1. Preprocess Data

```bash
python preprocessing/run.py -f best_conf/{dataset_name}.yml --dataset_name {dataset_name}
python preprocessing/to_nextpoi_kqt.py --dataset_name {dataset_name}
```

For the `ca` dataset, generate the raw format first:

```bash
python preprocessing/generate_ca_raw.py
```

### 2. Run the CoT Pipeline

```bash
cd cot
python 1-process_dataset.py --dataset_name {dataset_name}
python 2-generate_preference_async.py --dataset_name {dataset_name}
python 3-macro.py --dataset_name {dataset_name}
python 4_finetune_model.py --dataset_name {dataset_name}
python 4_run.py --dataset_name {dataset_name}
```

Optional fine-category evaluation:

```bash
python 4_test.py --dataset_name {dataset_name}
```

### 3. Build Final SFT Data

Return to the project root:

```bash
python merge_for_sft.py --dataset_name {dataset_name}
```

### 4. Fine-Tune the Final LLM

Example:

```bash
torchrun --nproc_per_node=1 supervised-fine-tune-qlora.py \
  --model_name_or_path model/Llama-2-7b-longlora-32k-ft \
  --bf16 True \
  --output_dir experiment/nyc \
  --model_max_length 32768 \
  --use_flash_attn True \
  --data_path datasets/nyc/final_sft_train.json \
  --eval_data_path datasets/nyc/final_sft_val.json \
  --low_rank_training True \
  --num_train_epochs 10 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --lr_scheduler_type constant_with_warmup \
  --logging_steps 10 \
  --deepspeed ds_configs/stage2.json \
  --tf32 True
```

### 5. Evaluate

```bash
python eval_next_poi.py \
  --model_path model/Llama-2-7b-longlora-32k-ft \
  --dataset_name nyc \
  --output_dir experiment/nyc/checkpoint-5500 \
  --test_file final_sft_test.json
```

## Ablation Study

The ablation workflow is documented in [ablation_study/README.md](ablation_study/README.md). It includes:

- CoT macro-stage ablations
- CoT fine-stage ablations
- final LLM input-signal ablations
- metrics collection and table generation

## Notes

- This repository contains code only. Datasets, model weights, generated preferences, and checkpoints should be prepared locally.
- API keys must stay in local environment variables or `.env`; they should never be committed.
- Training the final LLM stage is GPU-intensive. The NYC split is usually the fastest dataset for debugging.

## Acknowledgements

This project builds on ideas and implementation patterns from:

- LLM4POI: Large Language Models for Next Point-of-Interest Recommendation
- LongLoRA
- STHGCN

These works provided important references for long-context fine-tuning, POI recommendation, and spatio-temporal mobility modeling.
