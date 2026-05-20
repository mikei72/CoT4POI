from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import transformers
from peft import PeftModel
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from ablation_study.common.io import dump_json, load_json
from ablation_study.final_llm_ablation.prompting import PROMPT_TEMPLATE
from llama_attn_replace_sft import replace_llama_attn


def _extract_gt_id(text: str) -> str | None:
    match = re.search(r"(.*POI id\s*)(\d+)", text)
    if not match:
        return None
    return match.group(2)


def _extract_pred_id(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def load_generation_model(model_path: str, lora_output_dir: str, flash_attn: bool, context_size: int = 32768):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=context_size,
        padding_side="right",
        use_fast=True,
    )
    if flash_attn:
        replace_llama_attn(inference=True)

    config = transformers.AutoConfig.from_pretrained(model_path)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.resize_token_embeddings(32001)
    model.eval()

    if lora_output_dir:
        trainable_params = Path(lora_output_dir) / "trainable_params.bin"
        if trainable_params.is_file():
            model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        model = PeftModel.from_pretrained(
            model,
            lora_output_dir,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    return tokenizer, model


def evaluate_sft_json(
    data_path: Path,
    model_path: str,
    lora_output_dir: str,
    output_path: Path,
    num_return_sequences: int = 1,
    flash_attn: bool = True,
    reuse_existing_output: bool = True,
) -> dict:
    if reuse_existing_output and output_path.exists():
        cached = load_json(output_path)
        metrics = cached.get("metrics")
        if isinstance(metrics, dict) and "acc@1" in metrics:
            print(f"[final-llm-eval] Reuse cached evaluation: {output_path}")
            return metrics

    tokenizer, model = load_generation_model(model_path, lora_output_dir, flash_attn=flash_attn)
    data = load_json(data_path)
    correct_indices: list[int] = []
    total = 0
    correct = 0
    rows = []

    generation_config = transformers.GenerationConfig(
        max_new_tokens=5,
        do_sample=True,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.6,
        top_k=40,
        top_p=0.1,
        repetition_penalty=1.176,
        num_return_sequences=num_return_sequences,
    )

    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)

    for index, item in tqdm(enumerate(data), total=len(data), desc="final-llm-eval"):
        gt_text = item.get("target_answer", "")
        gt_id = _extract_gt_id(gt_text)
        if not gt_id:
            continue

        prompt_text = PROMPT_TEMPLATE.format(
            preference=item.get("system_preference", "No preference."),
            history=item.get("history_trajectory", ""),
            macro=item.get("aux_macro", "None"),
            fine=item.get("aux_fine", "None"),
            target_query=item.get("target_query_with_note", ""),
        )

        match = re.search(r"(.*POI id\s*)(\d+)", gt_text)
        prefix_text = match.group(1) if match else "POI id "
        final_input_text = prompt_text + " " + prefix_text

        prompt = tokenizer(final_input_text, return_tensors="pt").to(model.device)
        if prompt.input_ids.shape[1] >= 32768:
            continue

        outputs = model.generate(**prompt, generation_config=generation_config)
        pred_text = tokenizer.decode(outputs[:, prompt.input_ids.shape[1]:][0], skip_special_tokens=True)
        pred_id = _extract_pred_id(pred_text)
        is_correct = pred_id == gt_id
        total += 1
        if is_correct:
            correct += 1
            correct_indices.append(index)
        rows.append(
            {
                "index": index,
                "gt_id": gt_id,
                "pred_id": pred_id,
                "correct": is_correct,
            }
        )

    acc1 = (correct / total) if total > 0 else 0.0
    metrics = {
        "acc@1": round(float(acc1), 6),
        "total": int(total),
    }
    dump_json({"metrics": metrics, "correct_index": correct_indices, "details": rows}, output_path)
    return metrics
