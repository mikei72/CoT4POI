import json
import os
import re
from tqdm import tqdm
import argparse


def parse_txt_to_list(path):
    data_list = []
    print(f"Parsing TXT file: {path}...")

    orig_count = 0
    lines = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            orig_count += 1
            # 1. 仅针对特定的 "Caf+乱码" 进行替换，不使用 \W+ 这种可能包含换行符的通配符
            # 2. 保持行尾的换行符 \n 不动，确保行数不变
            processed_line = line.replace('Caf\ufffd\ufffd', 'Cafe').replace('Caf\x00\x00', 'Cafe')
            # 如果不知道具体乱码，用这个更安全的正则（不匹配换行符）
            import re
            processed_line = re.sub(r'Caf[^\w\s]+', 'Cafe', processed_line)

            lines.append(processed_line)

    for line in lines:
        line = line.strip()
        if not line: continue

        parts = line.split('<answer>:')
        if len(parts) < 2:
            continue

        q_part = parts[0].replace('<question>:', '').strip()

        a_part_raw = parts[1]
        if '<category>:' in a_part_raw:
            a_part = a_part_raw.split('<category>:')[0].strip()
        else:
            a_part = a_part_raw.strip()

        data_list.append({
            "question": q_part,
            "answer": a_part
        })
    return data_list


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_dict_to_str(d, top_k=10):
    if not d:
        return "None"
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ", ".join([f"{k} ({v:.2f})" for k, v in sorted_items])


def split_history_and_query(text):
    keyword = "Given the data,"
    if keyword in text:
        parts = text.split(keyword)
        history = parts[0].strip()
        query_with_note = keyword + parts[1]
        return history, query_with_note
    else:
        return text, ""


def process_single_split(config):
    orig_path = config['orig_path']
    sem_path = config['sem_path']
    out_path = config['out_path']

    if config['is_txt']:
        if not os.path.exists(orig_path):
            print(f"Warning: File {orig_path} not found. Skipping {config['split']}.\n")
            return
        orig_data = parse_txt_to_list(orig_path)
    else:
        orig_data = load_json(orig_path)

    sem_data = load_json(sem_path)

    if len(orig_data) != len(sem_data):
        print(f"Error: Length mismatch for {config['split']}! Orig: {len(orig_data)}, Sem: {len(sem_data)}\n")
        return

    merged_data = []
    print(f"Merging {config['split']} dataset ({len(orig_data)} items)...")

    for o_item, s_item in tqdm(zip(orig_data, sem_data), total=len(orig_data)):
        raw_q = o_item['question']
        if not config['is_txt']:  # JSON里的通常带前缀，TXT解析时已经去掉了
            raw_q = raw_q.replace("<question>:", "").strip()

        raw_a = o_item['answer'].replace("<answer>:", "").strip()

        history_str, query_str = split_history_and_query(raw_q)

        pref_str = s_item.get('preference', 'No specific preference.')
        macro_str = format_dict_to_str(s_item.get('macro', {}))
        fine_str = format_dict_to_str(s_item.get('fine', {}))

        merged_item = {
            "system_preference": pref_str,
            "history_trajectory": history_str,
            "aux_macro": macro_str,
            "aux_fine": fine_str,
            "target_query_with_note": query_str,  # 包含 "Given the data... Note..."
            "target_answer": raw_a
        }

        merged_data.append(merged_item)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {out_path}\n")


def main(dataset):
    files_config = [
        {
            "split": "train",
            "orig_path": f"datasets/{dataset}/preprocessed/train_qa_pairs_kqt.json",  # 原始 JSON
            "sem_path": f"datasets/{dataset}/dataIntegration/train_set_with_fine.json",  # 你的语义增强文件
            "out_path": f"datasets/{dataset}/preprocessed/final_sft_train.json",  # 输出文件
            "is_txt": False
        },
        {
            "split": "test",
            "orig_path": f"datasets/{dataset}/preprocessed/test_qa_pairs_kqt.txt",  # 原始 TXT
            "sem_path": f"datasets/{dataset}/dataIntegration/test_set_with_fine.json",
            "out_path": f"datasets/{dataset}/preprocessed/final_sft_test.json",
            "is_txt": True
        },
        {
            "split": "val",
            "orig_path": f"datasets/{dataset}/preprocessed/val_qa_pairs_kqt.txt",  # 原始 TXT (假设文件名)
            "sem_path": f"datasets/{dataset}/dataIntegration/val_set_with_fine.json",
            "out_path": f"datasets/{dataset}/preprocessed/final_sft_val.json",
            "is_txt": True
        }
    ]

    print("Starting Data Merge Pipeline...")
    for config in files_config:
        process_single_split(config)
    print("All Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")
    args = parser.parse_args()
    dataset = args.dataset_name

    main(dataset)