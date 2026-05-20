import torch
import random
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse

# 复用 utils
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
    "base_model": "BAAI/bge-reranker-base",
    # "base_model": f"../datasets/{dataset}/experiment/checkpoints/best_model",
    "train_file": f"../datasets/{dataset}/dataIntegration/train_set_with_macros.json",
    "val_file": f"../datasets/{dataset}/dataIntegration/val_set_with_macros.json",  # [NEW] 验证集路径
    "map_file": f"../datasets/{dataset}/dataIntegration/corrected_categories_with_macro.csv",
    "output_dir": f"../datasets/{dataset}/experiment/checkpoints",
    "max_len": 512,
    "epochs": 5,  # 稍微增加一点，反正有最佳模型保存策略
    "batch_size": 4,
    "lr": 2e-5,
    "neg_ratio": 7,
    "hard_neg_num": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "save_steps": 500,
    "patience": 5
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= 数据集定义 (保持不变) =================
class RerankDataset(Dataset):
    def __init__(self, data_list, macro2fine_dict, all_cats, is_validation=False):
        self.data = data_list
        self.macro2fine = macro2fine_dict
        self.all_cats = all_cats
        self.is_validation = is_validation  # [NEW] 标记是否为验证集

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = construct_prompt(item)

        raw_cat = item['result']['category_name']
        pos_cat = normalize_category(raw_cat)
        pos_cat_lower = pos_cat.lower()

        # --- 负采样逻辑 (训练和验证共用，验证集也需要负样本来计算分辨能力) ---
        raw_macros = item.get("macro", {})
        high_prob_macros = [m for m, p in raw_macros.items() if p > 0.3]

        hard_neg_candidates = []
        for m in high_prob_macros:
            m_key = m.lower().strip()
            if m_key in self.macro2fine:
                hard_neg_candidates.extend(self.macro2fine[m_key])

        hard_neg_pool = [c for c in hard_neg_candidates if c.lower().strip() != pos_cat_lower]
        hard_neg_pool = list(set(hard_neg_pool))

        easy_neg_pool = [c for c in self.all_cats if c.lower().strip() != pos_cat_lower]

        selected_negs = []

        # 验证集为了结果稳定，可以固定随机种子，或者这里不做特殊处理，
        # 因为Rerank能力的体现就是随机负样本能不能被压下去。
        num_hard = min(len(hard_neg_pool), CONFIG["hard_neg_num"])
        if num_hard > 0:
            selected_negs.extend(random.sample(hard_neg_pool, num_hard))

        num_easy = CONFIG["neg_ratio"] - len(selected_negs)
        if num_easy > 0:
            if len(easy_neg_pool) >= num_easy:
                selected_negs.extend(random.sample(easy_neg_pool, num_easy))
            else:
                selected_negs.extend(random.choices(easy_neg_pool, k=num_easy))

        # Pos 在第 0 位
        group_cats = [pos_cat] + selected_negs
        return query, group_cats


def collate_fn(batch):
    batch_queries = []
    batch_passages = []
    for query, cats in batch:
        for c in cats:
            batch_queries.append(query)
            batch_passages.append(c)
    return batch_queries, batch_passages


# ================= [NEW] 验证函数 =================
def evaluate(model, val_loader, tokenizer, device, steps):
    print(f"\nRunning Validation for steps {steps}...")
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0
    criterion = CrossEntropyLoss()

    with torch.no_grad():
        for queries, candidates in tqdm(val_loader, desc="Validating"):
            inputs = tokenizer(
                queries, candidates, padding=True, truncation=True,
                max_length=CONFIG["max_len"], return_tensors='pt'
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits  # [B * 8, 1]

            # Reshape: [Batch_Size, 8]
            group_size = CONFIG["neg_ratio"] + 1
            logits = logits.view(-1, group_size)

            # 标签全是 0 (因为Pos在第0位)
            labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)

            # 计算 Loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 计算 Accuracy (Hit@1 within batch)
            # 看每一行的最大值是不是在第 0 列
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()

            total_correct += correct
            total_samples += logits.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples

    print(f"Validation Result - Loss: {avg_loss:.4f} | Batch-Acc: {accuracy:.4f}\n")
    return accuracy  # 以此作为保存最佳模型的依据


# ================= 训练主流程 =================
def train():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1. 加载资源
    macro2fine = load_category_mapping(CONFIG["map_file"])
    all_cats = get_all_fine_categories(macro2fine)

    train_data = load_data(CONFIG["train_file"])
    val_data = load_data(CONFIG["val_file"])  # [NEW] 加载验证集

    print(f"Train Size: {len(train_data)} | Val Size: {len(val_data)}")

    # 2. 准备 DataLoader
    train_dataset = RerankDataset(train_data, macro2fine, all_cats, is_validation=False)
    val_dataset = RerankDataset(val_data, macro2fine, all_cats, is_validation=True)  # [NEW]

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn,
                              num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn,
                            num_workers=2)  # [NEW]

    # 3. 模型准备
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["base_model"])
    model.to(CONFIG["device"])

    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
    num_training_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * num_training_steps), num_training_steps)
    criterion = CrossEntropyLoss()

    # [NEW] 最佳模型记录
    best_acc = evaluate(model, val_loader, tokenizer, CONFIG["device"], -1)
    print(f"Initiative accuracy is {best_acc}\n\n")

    global_step = 0
    patience_counter = 0
    early_stop = False

    print("Start Training...")
    for epoch in range(CONFIG["epochs"]):
        # --- Train ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]")

        for queries, candidates in pbar:
            inputs = tokenizer(queries, candidates, padding=True, truncation=True, max_length=CONFIG["max_len"],
                               return_tensors='pt').to(CONFIG["device"])
            outputs = model(**inputs)
            logits = outputs.logits.view(-1, CONFIG["neg_ratio"] + 1)

            labels = torch.zeros(logits.size(0), dtype=torch.long).to(CONFIG["device"])
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % CONFIG["save_steps"] == 0:
                val_acc = evaluate(model, val_loader, tokenizer, CONFIG["device"], global_step)

                if val_acc > best_acc:
                    best_acc = val_acc
                    save_path = os.path.join(CONFIG["output_dir"], "best_model")
                    print(f"🔥 New Best Model (Acc: {best_acc:.4f})! Saving to {save_path}...\n\n")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                else:
                    patience_counter += 1
                    print(f"⚠️ No improvement. Patience: {patience_counter}/{CONFIG['patience']}\n")
                    if patience_counter >= CONFIG["patience"]:
                        print(f"🛑 Early stopping triggered at step {global_step}!")
                        early_stop = True
                        break  # 跳出当前的 batch 循环

                model.train()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if early_stop:
            print("训练已提前结束。")
            break

        save_epoch_path = os.path.join(CONFIG["output_dir"], f"epoch_{epoch+1}")
        os.makedirs(save_epoch_path, exist_ok=True)
        model.save_pretrained(save_epoch_path)
        tokenizer.save_pretrained(save_epoch_path)

    print(f"\nTraining Finished. Best Validation Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train()