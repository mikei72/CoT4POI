import pandas as pd
import json
from collections import Counter
import os
from typing import Dict, List, Any
import re
import argparse


def map_timestamp_to_discrete_features(timestamp: pd.Timestamp) -> (str, str):
    # --- 特征1: time_of_day (每日时段) ---
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
    else:  # 包含 21, 22, 23, 0 点
        time_of_day = "Late Night"

    # --- 特征2: day_type (周类型) ---
    weekday = timestamp.weekday()  # Monday=0, Sunday=6

    # 周末的定义：周五18点后，或者周六、周日一整天
    if (weekday == 4 and hour >= 18) or (weekday in [5, 6]):
        day_type = "Weekend"
    else:
        day_type = "Mid-Week"

    return time_of_day, day_type


def parse_single_sample(question: str, category_label: str) -> Dict[str, Any]:
    user_match = re.search(r"trajectory of user (\d+):", question)
    user_id = int(user_match.group(1)) if user_match else -1

    history_pattern = r"At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user \d+ visited POI id \d+ which is a (.*?) and has Category id \d+"
    history_matches = re.findall(history_pattern, question)

    history_list = []
    for time_str, cat_name in history_matches:
        dt = pd.to_datetime(time_str)
        time_of_day, day_type = map_timestamp_to_discrete_features(dt)

        history_list.append({
            "category_name": cat_name.strip(),
            "timestamp": dt.strftime('%Y-%m-%dT%H:%M:%S+00:00'),  # 格式化为 ISO 8601
            "time_of_day": time_of_day,
            "day_type": day_type
        })

    target_time_match = re.search(r"Given the data, At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),", question)

    if target_time_match:
        target_time_str = target_time_match.group(1)
        target_dt = pd.to_datetime(target_time_str)
        target_tod, target_dt_type = map_timestamp_to_discrete_features(target_dt)

        clean_target_cat = category_label.replace("<category>:", "").strip()

        result_obj = {
            "category_name": clean_target_cat,
            "timestamp": target_dt.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            "time_of_day": target_tod,
            "day_type": target_dt_type
        }
    else:
        # 如果匹配失败（通常不会发生，除非数据损坏）
        result_obj = {}

    return {
        "userId": user_id,
        "history": history_list,
        "result": result_obj
    }


def process_train_json(file_path):
    print(f"正在处理 JSON 文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        q = item.get("question", "")
        c = item.get("category", "")
        if q:
            processed_data.append(parse_single_sample(q, c))
    return processed_data


def process_txt_file(file_path):
    print(f"正在处理 TXT 文件: {file_path}")
    try:
        # 默认正常分支：直接按 utf-8 读取，不加任何替换逻辑，保证最高性能
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # 异常容错分支：仅在遇到无法解码的字符时进入
        # 使用 errors='replace' 会将无法识别的字节(如 0xa8) 变成特殊占位符 ''
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # 将乱码后的 Caf 以及可能存在的标准 Café 统一替换为 Cafe
        content = content.replace('Caf', 'Cafe').replace('Café', 'Cafe')

    raw_samples = content.split("<question>:")

    processed_data = []
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

        processed_data.append(parse_single_sample(question_part, category_part))

    return processed_data


# --- 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")
    args = parser.parse_args()

    dataset = args.dataset_name
    input_dir = f'../datasets/{dataset}/preprocessed'
    output_dir = f'../datasets/{dataset}/dataIntegration'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_input = os.path.join(input_dir, 'train_qa_pairs_kqt.json')
    if os.path.exists(train_input):
        print("开始处理Train集...")
        train_output = process_train_json(train_input)
        with open(os.path.join(output_dir, 'train_set.json'), 'w') as f:
            json.dump(train_output, f, indent=4)
        print(f"Train集处理完毕，样本数: {len(train_output)}")

    val_input = os.path.join(input_dir, 'val_qa_pairs_kqt.txt')
    if os.path.exists(val_input):
        print("开始处理Val集...")
        val_output = process_txt_file(val_input)
        with open(os.path.join(output_dir, 'val_set.json'), 'w') as f:
            json.dump(val_output, f, indent=4)
        print(f"Val集处理完毕，样本数: {len(val_output)}")

    test_input = os.path.join(input_dir, 'test_qa_pairs_kqt.txt')
    if os.path.exists(test_input):
        print("开始处理Test集...")
        test_output = process_txt_file(test_input)
        with open(os.path.join(output_dir, 'test_set.json'), 'w') as f:
            json.dump(test_output, f, indent=4)
        print(f"Test集处理完毕，样本数: {len(test_output)}")
