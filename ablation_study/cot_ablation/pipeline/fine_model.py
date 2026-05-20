from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ablation_study.common.cot_helpers import (
    construct_fine_prompt,
    get_all_fine_categories,
    load_category_mapping,
    normalize_category,
)
from ablation_study.common.io import dump_json, load_json
from ablation_study.common.metrics import format_metrics, ranking_metrics

FINE_INFER_SAVE_INTERVAL = 500
FINE_INFER_LOG_INTERVAL = 1000


@dataclass
class FineTrainConfig:
    base_model: str
    map_file: Path
    output_dir: Path
    max_len: int = 512
    epochs: int = 5
    batch_size: int = 4
    lr: float = 2e-5
    neg_ratio: int = 7
    hard_neg_num: int = 3
    save_steps: int = 500
    patience: int = 5
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RerankDataset(Dataset):
    def __init__(self, data_list: list[dict], macro2fine_dict: dict[str, list[str]], all_cats: list[str], use_preference: bool, use_macro: bool, neg_ratio: int, hard_neg_num: int):
        self.data = data_list
        self.macro2fine = macro2fine_dict
        self.all_cats = all_cats
        self.use_preference = use_preference
        self.use_macro = use_macro
        self.neg_ratio = neg_ratio
        self.hard_neg_num = hard_neg_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = construct_fine_prompt(item, use_preference=self.use_preference, use_macro=self.use_macro)
        pos_cat = normalize_category(item["result"]["category_name"])
        pos_cat_lower = pos_cat.lower()

        raw_macros = item.get("macro", {}) if self.use_macro else {}
        high_prob_macros = [m for m, p in raw_macros.items() if p > 0.3]

        hard_neg_candidates = []
        for macro_name in high_prob_macros:
            hard_neg_candidates.extend(self.macro2fine.get(macro_name.lower().strip(), []))
        hard_neg_pool = list({c for c in hard_neg_candidates if c.lower().strip() != pos_cat_lower})
        easy_neg_pool = [c for c in self.all_cats if c.lower().strip() != pos_cat_lower]

        selected_negs = []
        if hard_neg_pool:
            selected_negs.extend(random.sample(hard_neg_pool, min(len(hard_neg_pool), self.hard_neg_num)))
        needed_easy = self.neg_ratio - len(selected_negs)
        if needed_easy > 0:
            if len(easy_neg_pool) >= needed_easy:
                selected_negs.extend(random.sample(easy_neg_pool, needed_easy))
            else:
                selected_negs.extend(random.choices(easy_neg_pool, k=needed_easy))

        return query, [pos_cat] + selected_negs


def collate_fn(batch):
    queries = []
    passages = []
    for query, cats in batch:
        for cat in cats:
            queries.append(query)
            passages.append(cat)
    return queries, passages


def evaluate_group_accuracy(model, data_loader, tokenizer, device, group_size, max_len, steps: int):
    print(f"\nRunning Validation for steps {steps}...")
    model.eval()
    criterion = CrossEntropyLoss()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    with torch.no_grad():
        for queries, candidates in tqdm(data_loader, desc="fine-val", leave=True):
            inputs = tokenizer(queries, candidates, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            logits = model(**inputs).logits.view(-1, group_size)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total_samples += logits.size(0)
    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = total_correct / max(total_samples, 1)
    print(f"Validation Result - Loss: {avg_loss:.4f} | Batch-Acc: {accuracy:.4f}\n")
    return avg_loss, accuracy


def train_fine_model(train_file: Path, val_file: Path, use_preference: bool, use_macro: bool, config: FineTrainConfig) -> Path:
    set_seed(config.seed)
    macro2fine = load_category_mapping(config.map_file)
    all_cats = get_all_fine_categories(macro2fine)
    train_data = load_json(train_file)
    val_data = load_json(val_file)

    train_dataset = RerankDataset(train_data, macro2fine, all_cats, use_preference, use_macro, config.neg_ratio, config.hard_neg_num)
    val_dataset = RerankDataset(val_data, macro2fine, all_cats, use_preference, use_macro, config.neg_ratio, config.hard_neg_num)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(config.base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr)
    num_training_steps = max(len(train_loader) * config.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * num_training_steps), num_training_steps)
    criterion = CrossEntropyLoss()

    _, best_acc = evaluate_group_accuracy(
        model=model,
        data_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        group_size=config.neg_ratio + 1,
        max_len=config.max_len,
        steps=-1,
    )
    print(f"Initiative accuracy is {best_acc}\n")
    patience_counter = 0
    best_model_dir = config.output_dir / "best_model"
    os.makedirs(best_model_dir, exist_ok=True)
    global_step = 0
    early_stop = False

    for epoch in range(config.epochs):
        model.train()
        for queries, candidates in tqdm(train_loader, desc=f"fine-train-epoch-{epoch + 1}"):
            inputs = tokenizer(queries, candidates, padding=True, truncation=True, max_length=config.max_len, return_tensors="pt").to(device)
            logits = model(**inputs).logits.view(-1, config.neg_ratio + 1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % config.save_steps == 0:
                _, val_acc = evaluate_group_accuracy(
                    model=model,
                    data_loader=val_loader,
                    tokenizer=tokenizer,
                    device=device,
                    group_size=config.neg_ratio + 1,
                    max_len=config.max_len,
                    steps=global_step,
                )
                if val_acc > best_acc:
                    best_acc = val_acc
                    model.save_pretrained(best_model_dir)
                    tokenizer.save_pretrained(best_model_dir)
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        early_stop = True
                        break
                model.train()
        if early_stop:
            break

    if not (best_model_dir / "config.json").exists():
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
    return best_model_dir


def infer_fine_file(
    model_dir: Path,
    input_file: Path,
    output_file: Path,
    map_file: Path,
    use_preference: bool,
    use_macro: bool,
    target_candidate_size: int = 100,
    save_interval: int = FINE_INFER_SAVE_INTERVAL,
    log_interval: int = FINE_INFER_LOG_INTERVAL,
    infer_seed: int = 42,
) -> dict:
    random.seed(infer_seed)
    np.random.seed(infer_seed)
    torch.manual_seed(infer_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(infer_seed)

    macro2fine_dict = load_category_mapping(map_file)
    all_fine_categories = get_all_fine_categories(macro2fine_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(
        f"[fine-infer:{input_file.stem}] model_dir={model_dir}, device={device}, "
        f"tokenizer_class={tokenizer.__class__.__name__}, is_fast={getattr(tokenizer, 'is_fast', None)}, "
        f"infer_seed={infer_seed}"
    )

    test_data = load_json(input_file)
    if output_file.exists():
        existing = load_json(output_file)
        if len(existing) == len(test_data):
            restored = 0
            for idx, old_item in enumerate(existing):
                if old_item.get("fine"):
                    test_data[idx]["fine"] = old_item.get("fine")
                    restored += 1
            if restored > 0:
                print(f"[fine-infer:{input_file.stem}] resume from output, restored={restored}/{len(test_data)}")
        else:
            print(
                f"[fine-infer:{input_file.stem}] output length mismatch ({len(existing)} vs {len(test_data)}), "
                "ignore existing file and recompute."
            )

    ranks: list[int | None] = []
    start_time = time.time()
    processed = 0
    for idx, item in enumerate(tqdm(test_data, desc=f"fine:{input_file.stem}"), start=1):
        true_cat = normalize_category(item["result"]["category_name"])
        true_cat_lower = true_cat.lower()
        if not item.get("fine"):
            candidates = [true_cat]
            neg_pool = [c for c in all_fine_categories if c.lower().strip() != true_cat_lower]
            sample_size = min(len(neg_pool), target_candidate_size - 1)
            candidates.extend(random.sample(neg_pool, sample_size))
            random.shuffle(candidates)

            query = construct_fine_prompt(item, use_preference=use_preference, use_macro=use_macro)
            pairs = [[query, cand] for cand in candidates]
            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
                probs = torch.sigmoid(model(**inputs).logits.view(-1).float()).cpu().numpy()
            sorted_indices = np.argsort(probs)[::-1]
            predictions = [(candidates[p_idx], float(probs[p_idx])) for p_idx in sorted_indices]
            item["fine"] = {cat: score for cat, score in predictions[:10]}

        rank = None
        for r, cat in enumerate(list((item.get("fine") or {}).keys())):
            if str(cat).lower().strip() == true_cat_lower:
                rank = r
                break
        ranks.append(rank)
        processed += 1

        if log_interval > 0 and processed % log_interval == 0:
            elapsed = time.time() - start_time
            speed = processed / max(elapsed, 1e-9)
            print(
                f"[fine-infer:{input_file.stem}] progress={processed}/{len(test_data)}, "
                f"elapsed={elapsed/60:.2f}m, speed={speed:.2f} it/s"
            )

        if save_interval > 0 and processed % save_interval == 0:
            dump_json(test_data, output_file)
            print(f"[fine-infer:{input_file.stem}] checkpoint saved at {processed}/{len(test_data)}")

    dump_json(test_data, output_file)
    elapsed = time.time() - start_time
    print(
        f"[fine-infer:{input_file.stem}] finished total={len(test_data)}, "
        f"elapsed={elapsed/60:.2f}m, avg_speed={len(test_data)/max(elapsed, 1e-9):.2f} it/s"
    )
    return format_metrics(ranking_metrics(ranks, ks=(1, 5, 10)))
