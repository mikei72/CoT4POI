import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import random
import argparse

from utils.utils import (
    load_data,
    load_category_mapping,
    get_all_fine_categories,
    normalize_category,
    construct_prompt
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                    help="Name of the dataset (e.g., ca, nyc, tky)")
args = parser.parse_args()
dataset = args.dataset_name

CONFIG = {
    # "model_name": "BAAI/bge-reranker-base",  # 本地或者 HuggingFace Hub 路径
    "model_name": f"../datasets/{dataset}/experiment/checkpoints/best_model",
    # "model_name": "experiment/checkpoints/step_3501",
    "data_file": f"../datasets/{dataset}/dataIntegration/test_set_with_macros.json",
    "map_file": f"../datasets/{dataset}/dataIntegration/corrected_categories_with_macro.csv",
    "batch_size": 1,  # 推理时的 batch size (指处理多少个 user session)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_detail_count": 3  # 详细输出前 N 个片段的预测结果
}


def main():
    # 1. 资源加载
    macro2fine_dict = load_category_mapping(CONFIG["map_file"])
    test_data = load_data(CONFIG["data_file"])
    all_fine_categories = get_all_fine_categories(macro2fine_dict)
    print(f"Total unique fine categories: {len(all_fine_categories)}")

    print(f"Loading Model: {CONFIG['model_name']} to {CONFIG['device']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"])
    model.to(CONFIG["device"])
    model.eval()

    # 指标统计
    metrics = {
        "hit1": 0, "hit5": 0, "hit10": 0, "mrr": 0, "total": 0
    }

    print("Starting Inference...")
    print("=" * 60)

    # 2. 推理循环
    for i, item in enumerate(tqdm(test_data)):
        # ---------------- A. Ground Truth 处理 ----------------
        raw_cat = item['result']['category_name']
        true_cat = normalize_category(raw_cat)
        true_cat_lower = true_cat.lower()

        # ---------------- B. 动态候选集生成 ----------------
        final_candidates = [true_cat]
        # 从全局池中过滤掉 GT
        neg_pool = [c for c in all_fine_categories if c.lower().strip() != true_cat_lower]

        target_size = 100
        num_to_sample = target_size - 1

        if len(neg_pool) <= num_to_sample:
            final_candidates.extend(neg_pool)
        else:
            final_candidates.extend(random.sample(neg_pool, num_to_sample))

        random.shuffle(final_candidates)

        # ---------------- C. Prompt 构造 ----------------
        query = construct_prompt(item)

        # ---------------- D. BGE-Reranker 推理 ----------------
        # 构造 Pairs: [[Query, Cand1], [Query, Cand2], ...]
        pairs = [[query, cand] for cand in final_candidates]

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(CONFIG["device"])

            scores = model(**inputs).logits.view(-1).float()
            probs = torch.sigmoid(scores).cpu().numpy()

        # ---------------- E. 排序与评估 ----------------
        sorted_indices = np.argsort(probs)[::-1]

        rank = -1
        pred_list_for_log = []

        for r, idx in enumerate(sorted_indices):
            cand_name = final_candidates[idx]

            if r < 10:
                pred_list_for_log.append((cand_name, probs[idx]))

            if cand_name.lower().strip() == true_cat_lower:
                rank = r

        # 统计
        metrics["total"] += 1
        if rank == 0: metrics["hit1"] += 1
        if rank < 5 and rank != -1: metrics["hit5"] += 1
        if rank < 10 and rank != -1: metrics["hit10"] += 1
        if rank != -1:
            metrics["mrr"] += 1.0 / (rank + 1)

        # ---------------- F. 详细日志 (前 N 个) ----------------
        if i < CONFIG["log_detail_count"]:
            print(f"\n[Session {i}] Ground Truth: {true_cat}")
            print(f"Query: {query}")

            print("-" * 20 + " Top 10 Predictions " + "-" * 20)
            found_in_top10 = False
            for rank_idx, (c_name, c_score) in enumerate(pred_list_for_log):
                mark = "✅ HIT" if c_name.lower().strip() == true_cat_lower else ""
                if mark: found_in_top10 = True
                print(f"{rank_idx + 1}. [{c_score:.4f}] {c_name} {mark}")

            if not found_in_top10 and rank != -1:
                print(f"... GT found at Rank {rank + 1}")
            elif rank == -1:
                print("!! GT not found in candidates (Logic Error?) !!")
            print("=" * 60)

    # 3. 最终结果输出
    print("\n" + "=" * 20 + " Final Results " + "=" * 20)
    total = metrics["total"]
    print(f"Total Sessions: {total}")
    print(f"Hit@1 : {metrics['hit1'] / total:.4f}")
    print(f"Hit@5 : {metrics['hit5'] / total:.4f}")
    print(f"Hit@10: {metrics['hit10'] / total:.4f}")
    print(f"MRR   : {metrics['mrr'] / total:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()