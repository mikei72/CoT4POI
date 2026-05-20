from __future__ import annotations

import csv
import io
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MACRO_CATEGORIES = [
    "food & dining",
    "arts & entertainment",
    "shopping & retail",
    "health & wellness",
    "travel & transportation",
    "professional & public services",
    "outdoors & nature",
]


def map_timestamp_to_discrete_features(timestamp: pd.Timestamp) -> tuple[str, str]:
    hour = timestamp.hour
    if 1 <= hour < 5:
        time_of_day = "Pre-dawn"
    elif 5 <= hour < 9:
        time_of_day = "Morning"
    elif 9 <= hour < 12:
        time_of_day = "Forenoon"
    elif 12 <= hour < 14:
        time_of_day = "Noon"
    elif 14 <= hour < 18:
        time_of_day = "Afternoon"
    elif 18 <= hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Late Night"

    weekday = timestamp.weekday()
    if (weekday == 4 and hour >= 18) or weekday in [5, 6]:
        day_type = "Weekend"
    else:
        day_type = "Mid-Week"
    return time_of_day, day_type


def normalize_category(cat_name: str) -> str:
    if not cat_name:
        return ""
    if cat_name == "Café":
        return "Cafe"
    return cat_name.strip()


def parse_single_sample(question: str, category_label: str, use_time_discretization: bool = True) -> dict[str, Any]:
    user_match = re.search(r"trajectory of user (\d+):", question)
    user_id = int(user_match.group(1)) if user_match else -1

    history_pattern = (
        r"At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user \d+ visited POI id \d+ which is a "
        r"(.*?) and has Category id \d+"
    )
    history_matches = re.findall(history_pattern, question)

    history_list = []
    for time_str, cat_name in history_matches:
        dt = pd.to_datetime(time_str)
        time_of_day, day_type = map_timestamp_to_discrete_features(dt)
        item = {
            "category_name": cat_name.strip(),
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        }
        if use_time_discretization:
            item["time_of_day"] = time_of_day
            item["day_type"] = day_type
        history_list.append(item)

    target_time_match = re.search(r"Given the data, At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),", question)
    result_obj: dict[str, Any] = {}
    if target_time_match:
        target_dt = pd.to_datetime(target_time_match.group(1))
        result_obj = {
            "category_name": category_label.replace("<category>:", "").strip(),
            "timestamp": target_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        }
        if use_time_discretization:
            target_tod, target_day_type = map_timestamp_to_discrete_features(target_dt)
            result_obj["time_of_day"] = target_tod
            result_obj["day_type"] = target_day_type
    return {"userId": user_id, "history": history_list, "result": result_obj}


def process_train_json(file_path: Path, use_time_discretization: bool = True) -> list[dict]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    processed = []
    for item in data:
        q = item.get("question", "")
        c = item.get("category", "")
        if q:
            processed.append(parse_single_sample(q, c, use_time_discretization=use_time_discretization))
    return processed


def process_txt_file(file_path: Path, use_time_discretization: bool = True) -> list[dict]:
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        content = content.replace("Caf", "Cafe").replace("Café", "Cafe")

    raw_samples = content.split("<question>:")
    processed = []
    for raw in raw_samples:
        if not raw.strip():
            continue
        full_text = "<question>:" + raw
        parts_answer = full_text.split("<answer>:")
        if len(parts_answer) < 2:
            continue
        question_part = parts_answer[0]
        rest = parts_answer[1]
        parts_category = rest.split("<category>:")
        if len(parts_category) < 2:
            continue
        category_part = parts_category[1].strip()
        processed.append(parse_single_sample(question_part, category_part, use_time_discretization=use_time_discretization))
    return processed


def construct_preference_prompt(history_data: list, target_day_type: str, target_time_of_day: str, specific_time: str) -> tuple[str, str]:
    def _history_item(item: dict) -> str:
        if "day_type" in item and "time_of_day" in item:
            return f"[{item['day_type']} {item['time_of_day']}] {item['category_name']}"
        return item["category_name"]

    user_history_str = " -> ".join(_history_item(item) for item in history_data)
    system_prompt = (
        "You are an expert in human behavioral dynamics. Your task is to infer a user's next "
        "macro-level behavioral intention based on their historical trajectory and a specific time context."
    )

    if target_day_type and target_time_of_day:
        user_prompt = f"""
        [User History]: {user_history_str}
        [Current Target Time]: {target_day_type} {target_time_of_day} ({specific_time})

        [Analytical Framework]: To formulate your response, please consider the following analytical steps internally:
            1.  **Thematic History Analysis**: What is the dominant theme or pattern in the user's history (e.g., 'health-conscious', 'work-focused', 'socially active')?
            2.  **Contextual Contrast**: How does this historical theme contrast with the general public's typical behavior at the [Current Target Time]? Highlight any conflict or alignment (e.g., "The user's disciplined weekday pattern conflicts with a typical relaxed weekend evening").
            3.  **Inference of Intentional Shift**: Based on the contrast, infer the user's likely psychological shift. Are they likely to continue their pattern or break from it?

        [Output Format]: Synthesize your analysis from the framework above into a **single, fluid paragraph**, under 50 words.
            - **Do not** use bullet points, numbered lists, or labels like "Step 1".
            - The tone should be that of an expert providing a concise, narrative summary.
            - Start by acknowledging the user's pattern and the current context.
            - Conclude by stating the inferred intention and the most likely macro-categories of interest (e.g., Food, Entertainment, Shopping).
        """
    else:
        user_prompt = f"""
        [User History]: {user_history_str}
        [Current Target Time]: {specific_time}

        [Analytical Framework]: To formulate your response, please consider the following analytical steps internally:
            1.  **Thematic History Analysis**: What is the dominant theme or pattern in the user's history?
            2.  **Inference of Intentional Shift**: Infer the likely next intention without discrete temporal labels.

        [Output Format]: Synthesize your analysis from the framework above into a **single, fluid paragraph**, under 50 words.
            - **Do not** use bullet points, numbered lists, or labels like "Step 1".
            - Conclude by stating the inferred intention and the most likely macro-categories of interest.
        """
    return system_prompt, user_prompt


def history_to_macro_text(item: dict, use_preference: bool) -> str:
    if use_preference and item.get("preference"):
        return item["preference"]

    parts = []
    for hist in item.get("history", []):
        if "day_type" in hist and "time_of_day" in hist:
            parts.append(f"{hist['category_name']} ({hist['day_type']} {hist['time_of_day']})")
        else:
            parts.append(hist["category_name"])

    if "day_type" in item.get("result", {}) and "time_of_day" in item["result"]:
        parts.append(f"target context: {item['result']['day_type']} {item['result']['time_of_day']}")
    return " -> ".join(parts)


def load_category_mapping(csv_path: Path) -> dict[str, list[str]]:
    macro2fine: dict[str, list[str]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2 or row[0] == "CategoryName":
                continue
            fine_cat = row[0].strip()
            macro_cat = row[-1].strip().lower()
            macro2fine.setdefault(macro_cat, []).append(fine_cat)
    return macro2fine


def get_all_fine_categories(macro2fine_dict: dict[str, list[str]]) -> list[str]:
    values: list[str] = []
    for cat_list in macro2fine_dict.values():
        values.extend(cat_list)
    return sorted(set(values))


def construct_fine_prompt(item: dict, use_preference: bool = True, use_macro: bool = True) -> str:
    history_str_list = []
    for hist in item.get("history", []):
        if "day_type" in hist and "time_of_day" in hist:
            history_str_list.append(f"{hist['category_name']} ({hist['day_type']} {hist['time_of_day']})")
        else:
            history_str_list.append(hist["category_name"])
    history_seq = " -> ".join(history_str_list) if history_str_list else "Empty History"

    target = item.get("result", {})
    if "day_type" in target and "time_of_day" in target:
        target_time = f"{target['day_type']} {target['time_of_day']}"
    else:
        target_time = target.get("timestamp", "Unknown")

    pref_text = item.get("preference", "No specific preference.") if use_preference else "No specific preference."
    macro_hint = ", ".join(item.get("macro", {})) if use_macro else "None"

    return (
        f"[User History]: {history_seq}\n"
        f"[Target Context]: {target_time}\n"
        f"[Intent Analysis]: {pref_text}\n"
        f"[Search categories]: {macro_hint}\n"
        "Based on the above, next the user is most likely to visit:"
    )


def format_dict_to_str(d: dict[str, float] | None, top_k: int = 10) -> str:
    if not d:
        return "None"
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ", ".join([f"{k} ({v:.2f})" for k, v in sorted_items])


def split_history_and_query(text: str) -> tuple[str, str]:
    keyword = "Given the data,"
    if keyword in text:
        parts = text.split(keyword)
        history = parts[0].strip()
        query_with_note = keyword + parts[1]
        return history, query_with_note
    return text, ""


def parse_txt_to_list(path: Path) -> list[dict]:
    data_list = []
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            processed_line = line.replace("Caf\ufffd\ufffd", "Cafe").replace("Caf\x00\x00", "Cafe")
            processed_line = re.sub(r"Caf[^\w\s]+", "Cafe", processed_line)
            lines.append(processed_line)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("<answer>:")
        if len(parts) < 2:
            continue
        q_part = parts[0].replace("<question>:", "").strip()
        a_part_raw = parts[1]
        if "<category>:" in a_part_raw:
            a_part = a_part_raw.split("<category>:")[0].strip()
        else:
            a_part = a_part_raw.strip()
        data_list.append({"question": q_part, "answer": a_part})
    return data_list


def load_raw_split(dataset_root: Path, split: str) -> list[dict]:
    preprocessed_dir = dataset_root / "preprocessed"
    if split == "train":
        path = preprocessed_dir / "train_qa_pairs_kqt.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    path = preprocessed_dir / f"{split}_qa_pairs_kqt.txt"
    return parse_txt_to_list(path)


def build_sft_examples(orig_data: list[dict], sem_data: list[dict], include_preference: bool, include_macro: bool, include_fine: bool) -> list[dict]:
    if len(orig_data) != len(sem_data):
        raise ValueError(f"Length mismatch: raw={len(orig_data)} sem={len(sem_data)}")

    merged_data = []
    for o_item, s_item in zip(orig_data, sem_data):
        raw_q = o_item["question"]
        if raw_q.startswith("<question>:"):
            raw_q = raw_q.replace("<question>:", "", 1).strip()
        raw_a = o_item["answer"].replace("<answer>:", "").strip()
        history_str, query_str = split_history_and_query(raw_q)

        merged_item = {
            "system_preference": s_item.get("preference", "No specific preference.") if include_preference else "No specific preference.",
            "history_trajectory": history_str,
            "aux_macro": format_dict_to_str(s_item.get("macro", {})) if include_macro else "None",
            "aux_fine": format_dict_to_str(s_item.get("fine", {})) if include_fine else "None",
            "target_query_with_note": query_str,
            "target_answer": raw_a,
        }
        merged_data.append(merged_item)
    return merged_data


def read_all_category_counts(raw_dataset_root: Path, dataset_name: str) -> list[dict]:
    if dataset_name == "nyc":
        frames = [
            pd.read_csv(raw_dataset_root / "raw" / "NYC_train.csv"),
            pd.read_csv(raw_dataset_root / "raw" / "NYC_val.csv"),
            pd.read_csv(raw_dataset_root / "raw" / "NYC_test.csv"),
        ]
        df = pd.concat(frames, ignore_index=True)
        target_column = "PoiCategoryName"
    elif dataset_name == "ca":
        df = pd.read_csv(raw_dataset_root / "raw" / "dataset_gowalla_ca_ne.csv")
        target_column = "PoiCategoryId"
    elif dataset_name == "tky":
        file_path = raw_dataset_root / "raw" / "dataset_TSMC2014_TKY.txt"
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            content = content.replace("Caf\ufffd", "Cafe").replace("Café", "Cafe")
        data_buffer = io.StringIO(content)
        df = pd.read_csv(data_buffer, sep="\t", header=None)
        df.columns = ["UserId", "PoiId", "CategoryHash", "PoiCategoryId", "Latitude", "Longitude", "TimezoneOffset", "UTCTime"]
        target_column = "PoiCategoryId"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    category_counts = df[target_column].value_counts().reset_index()
    category_counts.columns = ["CategoryName", "Count"]
    return category_counts.to_dict(orient="records")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
