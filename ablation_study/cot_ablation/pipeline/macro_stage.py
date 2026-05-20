from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline

from ablation_study.common.cot_helpers import MACRO_CATEGORIES, history_to_macro_text
from ablation_study.common.io import dump_json, load_json
from ablation_study.common.metrics import format_metrics, ranking_metrics

MACRO_SAVE_INTERVAL = 500
MACRO_PROGRESS_INTERVAL = 200
MAX_MACRO_TEXT_CHARS = 4000


def initialize_classifier(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    print(
        f"[macro] initialize classifier model={model_name}, "
        f"cuda_available={torch.cuda.is_available()}, device={device}"
    )
    clf = pipeline("zero-shot-classification", model=model_name, device=device)
    print(f"[macro] classifier ready, framework={getattr(clf, 'framework', 'unknown')}")
    return clf


def classify_macro_distribution(classifier, text: str, threshold: float = 0.2, min_items: int = 2) -> dict[str, float]:
    if not text:
        return {}
    if len(text) > MAX_MACRO_TEXT_CHARS:
        text = text[:MAX_MACRO_TEXT_CHARS]
    result = classifier(text, MACRO_CATEGORIES, multi_label=True)
    scored = sorted(zip(result["labels"], result["scores"]), key=lambda x: x[1], reverse=True)
    kept = {label.lower(): round(score, 4) for label, score in scored if score > threshold}
    if len(kept) >= min_items:
        return kept
    return {label.lower(): round(score, 4) for label, score in scored[:min_items]}


def run_macro_file(classifier, input_path: Path, output_path: Path, use_preference: bool) -> dict:
    data = load_json(input_path)
    if output_path.exists():
        existing = load_json(output_path)
        if len(existing) == len(data):
            for idx, old_item in enumerate(existing):
                if "macro" in old_item:
                    data[idx]["macro"] = old_item.get("macro")
                if old_item.get("macro_error"):
                    data[idx]["macro_error"] = old_item.get("macro_error")
            done = sum(1 for item in data if "macro" in item)
            print(f"[macro:{input_path.stem}] resume from {output_path}, already_done={done}/{len(data)}")
        else:
            print(
                f"[macro:{input_path.stem}] warning: output length mismatch "
                f"({len(existing)} vs {len(data)}), ignore existing file and recompute."
            )

    ranks: list[int | None] = []
    processed = 0
    failed = 0
    for idx, item in enumerate(tqdm(data, desc=f"macro:{input_path.stem}"), start=1):
        if idx % MACRO_PROGRESS_INTERVAL == 0:
            print(f"[macro:{input_path.stem}] progress idx={idx}/{len(data)}")

        if "macro" not in item:
            text = history_to_macro_text(item, use_preference=use_preference)
            try:
                macro_dict = classify_macro_distribution(classifier, text=text)
                item["macro"] = macro_dict
                item.pop("macro_error", None)
            except Exception as exc:
                item["macro"] = {}
                item["macro_error"] = str(exc)
                failed += 1
                print(
                    f"[macro:{input_path.stem}] error at idx={idx}, text_len={len(text)}: {exc}"
                )

        macro_dict = item.get("macro", {}) or {}
        true_macro = item.get("result", {}).get("macro_category")
        rank = None
        if true_macro:
            labels = list(macro_dict.keys())
            try:
                rank = labels.index(true_macro.lower())
            except ValueError:
                rank = None
        ranks.append(rank)
        processed += 1

        if processed % MACRO_SAVE_INTERVAL == 0:
            dump_json(data, output_path)
            print(
                f"[macro:{input_path.stem}] checkpoint saved at {processed}/{len(data)}, failed={failed}"
            )

    dump_json(data, output_path)
    print(f"[macro:{input_path.stem}] finished total={len(data)}, failed={failed}, output={output_path}")
    return format_metrics(ranking_metrics(ranks, ks=(1, 3, 5)))


def evaluate_existing_macro_file(path: Path) -> dict:
    data = load_json(path)
    ranks: list[int | None] = []
    for item in data:
        macro_dict = item.get("macro", {}) or {}
        labels = list(macro_dict.keys())
        true_macro = item.get("result", {}).get("macro_category")
        rank = None
        if true_macro:
            try:
                rank = labels.index(str(true_macro).lower())
            except ValueError:
                rank = None
        ranks.append(rank)
    return format_metrics(ranking_metrics(ranks, ks=(1, 3, 5)))
