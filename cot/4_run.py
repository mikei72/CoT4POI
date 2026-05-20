import json
from textwrap import indent

import torch
import numpy as np
import random
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse

# 复用工具函数
from utils.utils import (
    load_data,
    load_category_mapping,
    get_all_fine_categories,
    normalize_category,
    construct_prompt
)

# ================= 配置区域 =================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                    help="Name of the dataset (e.g., ca, nyc, tky)")
args = parser.parse_args()
dataset = args.dataset_name

CONFIG = {
    "model_name": f"../datasets/{dataset}/experiment/checkpoints/best_model",
    "map_file": f"../datasets/{dataset}/dataIntegration/corrected_categories_with_macro.csv",
    "dataset_path": f"../datasets/{dataset}/dataIntegration/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "target_candidate_size": 100,
    "score_threshold": 0.5,
    "min_result_count": 3,
    "max_result_count": 10,
    "save_steps": 500
}


def process_dataset(model, tokenizer, all_fine_categories, input_file, output_file):
    print(f"\nProcessing: {input_file} -> {output_file}")

    data_list = load_data(input_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        print("发现已有输出文件，正在加载进度...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            # 将已有的 fine 结果同步到当前的 data_list
            for i in range(min(len(data_list), len(existing_data))):
                if "fine" in existing_data[i]:
                    data_list[i]["fine"] = existing_data[i]["fine"]
        except Exception as e:
            print(f"加载进度失败，将重新开始: {e}")

    for i, item in enumerate(tqdm(data_list, desc="Inference")):
        if "fine" in item:
            continue

        raw_cat = item['result']['category_name']
        true_cat = normalize_category(raw_cat)
        true_cat_lower = true_cat.lower()

        final_candidates = [true_cat]
        neg_pool = [c for c in all_fine_categories if c.lower().strip() != true_cat_lower]

        target_size = CONFIG["target_candidate_size"]
        if len(neg_pool) <= (target_size - 1):
            final_candidates.extend(neg_pool)
        else:
            final_candidates.extend(random.sample(neg_pool, target_size - 1))
        random.shuffle(final_candidates)

        query = construct_prompt(item)

        pairs = [[query, cand] for cand in final_candidates]

        with torch.no_grad():
            inputs = tokenizer(
                pairs, padding=True, truncation=True, return_tensors='pt', max_length=512
            ).to(CONFIG["device"])

            scores = model(**inputs).logits.view(-1).float()
            probs = torch.sigmoid(scores).cpu().numpy()

        sorted_indices = np.argsort(probs)[::-1]

        all_predictions = []
        for idx in sorted_indices:
            cat = final_candidates[idx]
            score = float(probs[idx])
            all_predictions.append((cat, score))

        threshold = CONFIG["score_threshold"]
        filtered_preds = [p for p in all_predictions if p[1] > threshold]

        if len(filtered_preds) < CONFIG["min_result_count"]:
            final_preds_list = all_predictions[:CONFIG["min_result_count"]]
        elif len(filtered_preds) > CONFIG["max_result_count"]:
            final_preds_list = filtered_preds[:CONFIG["max_result_count"]]
        else:
            final_preds_list = filtered_preds

        fine_dict = {cat: score for cat, score in final_preds_list}

        item["fine"] = fine_dict

        if (i + 1) % CONFIG["save_steps"] == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=4, ensure_ascii=False)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(data_list)} items to {output_file}")


def main():
    print(f"Loading Resources...")
    macro2fine_dict = load_category_mapping(CONFIG["map_file"])
    all_fine_categories = get_all_fine_categories(macro2fine_dict)

    print(f"Loading Model: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"])
    model.to(CONFIG["device"])
    model.eval()

    files_to_process = [
        (
            f"{CONFIG['dataset_path']}/test_set_with_macros.json",
            f"{CONFIG['dataset_path']}/test_set_with_fine.json"
        ),
        (
            f"{CONFIG['dataset_path']}/val_set_with_macros.json",
            f"{CONFIG['dataset_path']}/val_set_with_fine.json"
        ),
        (
            f"{CONFIG['dataset_path']}/train_set_with_macros.json",
            f"{CONFIG['dataset_path']}/train_set_with_fine.json"
        )
    ]

    print("=" * 60)
    print(f"Starting Batch Processing. Total files: {len(files_to_process)}")

    for in_file, out_file in files_to_process:
        process_dataset(model, tokenizer, all_fine_categories, in_file, out_file)

    print("=" * 60)
    print("All tasks finished!")


if __name__ == "__main__":
    main()