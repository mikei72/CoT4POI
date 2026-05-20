import json
import time
from typing import Dict, List, Any
from tqdm import tqdm
from transformers import pipeline
import torch
import argparse

from utils.distribution import get_distribution
from utils.generate_category_map import create_fine_to_macro_map

# --- 1. Configuration ---
MODEL_NAME = "cross-encoder/nli-roberta-base"
MACRO_CATEGORIES = [
    "food & dining",
    "arts & entertainment",
    "shopping & retail",
    "health & wellness",
    "travel & transportation",
    "professional & public services",
    "outdoors & nature"
]
# 设置每处理多少个条目就保存并打印一次报告
SAVE_INTERVAL = 500


# --- 2. Model Initialization Function ---
def initialize_classifier(model_name: str) -> Any:
    print(f"Initializing zero-shot classification model: '{model_name}'...")
    try:
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            print("CUDA (GPU) detected. Loading model on GPU for faster processing.")
        else:
            print("No GPU detected. Loading model on CPU.")
        classifier = pipeline("zero-shot-classification", model=model_name, device=device)
        print("Model loaded successfully!")
        return classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# --- 3. Core Logic for a Single Prediction ---
def get_macro_distribution(classifier: Any, preference_text: str, categories: List[str], threshold: float = 0.2,
                           min_items: int = 2) -> Dict[str, float]:
    if not preference_text:
        return {}
    try:
        result = classifier(preference_text, categories, multi_label=True)
        scored_labels = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)
        macro_distribution = {
            label.lower(): round(score, 4)
            for label, score in scored_labels
            if score > threshold
        }

        if len(macro_distribution) < min_items:
            top_n_items = scored_labels[:min_items]
            final_distribution = {
                label.lower(): round(score, 4)
                for label, score in top_n_items
            }
            return final_distribution
        else:
            return macro_distribution
    except Exception as e:
        tqdm.write(f"\nError during classification for a text entry: {e}")
        return {"error": "ClassificationFailed"}


# --- 4. Main Batch Processing Function (Serial Version) ---
def batch_add_macro_predictions_serial(classifier: Any, input_filepath: str, output_filepath: str):
    try:
        with open(output_filepath, "r", encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded existing progress from '{output_filepath}'.")
    except (FileNotFoundError, json.JSONDecodeError):
        with open(input_filepath, "r", encoding='utf-8') as f:
            data = json.load(f)
        print(f"Starting fresh from original file '{input_filepath}'.")

    # Filter out items that need processing
    items_to_process_indices = [i for i, entry in enumerate(data) if not entry.get("macro")]
    total_to_process = len(items_to_process_indices)

    if total_to_process == 0:
        print("All entries already have macro predictions. Nothing to do.")
        return

    print(f"Found {total_to_process} entries to process.")

    # 【核心修改】初始化计时器和计数器
    total_start_time = time.time()
    batch_start_time = time.time()
    processed_count = 0

    # 使用 tqdm 包装需要处理的索引列表
    for i, entry_index in enumerate(tqdm(items_to_process_indices, desc="Classifying Preferences")):
        entry = data[entry_index]
        if entry.get('macro'):
            continue

        preference_text = entry.get("preference")

        # --- 串行调用 ---
        macro_dict = get_macro_distribution(classifier, preference_text, MACRO_CATEGORIES)
        entry['macro'] = macro_dict

        # --- 检查是否达到了保存和报告的阈值 ---
        # i 从 0 开始，所以 processed_count 是 i + 1
        processed_count += 1

        if (i + 1) % SAVE_INTERVAL == 0 or i == total_to_process - 1:
            # --- Calculate Times ---
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            total_elapsed = batch_end_time - total_start_time

            avg_time_per_item = total_elapsed / processed_count
            remaining_items = total_to_process - (i + 1)
            eta = remaining_items * avg_time_per_item

            # --- Print Report ---
            tqdm.write("-" * 50)
            tqdm.write(f"批次报告 ({i + 1}/{total_to_process}):")
            tqdm.write(f"  本批 ({SAVE_INTERVAL}条或最后一批) 耗时: {batch_time:.2f}s")
            tqdm.write(f"  总计已运行: {total_elapsed:.2f}s")
            tqdm.write(f"  预计剩余时间: {eta:.2f}s (~{eta / 60:.2f} 分钟)")

            # --- Save Progress ---
            with open(output_filepath, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            tqdm.write(f"  进度已保存到 '{output_filepath}'")
            tqdm.write("-" * 50)

            # --- Reset Batch Timer ---
            batch_start_time = time.time()

    print("\n所有条目处理和保存完成！")


def main(file, classifier):
    output_path = file.replace(".json", "_with_macros.json")

    batch_add_macro_predictions_serial(
        classifier=classifier,
        input_filepath=file,
        output_filepath=output_path
    )

# --- 5. Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")
    args = parser.parse_args()
    dataset = args.dataset_name

    classifier = initialize_classifier(MODEL_NAME)
    if classifier:
        main(f'../datasets/{dataset}/dataIntegration/test_set.json', classifier)
        main(f'../datasets/{dataset}/dataIntegration/val_set.json', classifier)
        main(f'../datasets/{dataset}/dataIntegration/train_set.json', classifier)
    else:
        print("Error: invalid classifier")

    get_distribution(dataset)
    create_fine_to_macro_map(dataset)

