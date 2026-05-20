import json
import csv
import random


def load_data(json_path):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} session items.")
    return data


def load_category_mapping(csv_path):
    print(f"Loading mapping from {csv_path}...")
    macro2fine = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            fine_cat = row[0].strip()
            macro_cat = row[-1].strip().lower()  # 统一小写作为 Key

            if macro_cat not in macro2fine:
                macro2fine[macro_cat] = []
            macro2fine[macro_cat].append(fine_cat)
    print(f"Loaded {len(macro2fine)} macro categories.")
    return macro2fine


def get_all_fine_categories(macro2fine_dict):
    all_cats = []
    for cat_list in macro2fine_dict.values():
        all_cats.extend(cat_list)
    return list(set(all_cats))


def normalize_category(cat_name):
    if not cat_name:
        return ""

    if cat_name == "Café":
        cat_name = "Cafe"
    return cat_name.strip()


def construct_prompt(item):
    history_list = item.get('history', [])
    history_str_list = []
    for h in history_list:
        node = f"{h['category_name']} ({h['day_type']} {h['time_of_day']})"
        history_str_list.append(node)

    # 用箭头连接，体现时序流动
    history_seq = " -> ".join(history_str_list) if history_str_list else "Empty History"

    # 2. 当前时空上下文 (Target Context)
    target_time = f"{item['result']['day_type']} {item['result']['time_of_day']}"

    # 3. 偏好与意图 (Preference)
    pref_text = item.get("preference", "No specific preference.")

    # 4. 宏观指引 (Macro Hints)
    macro_hint = ", ".join(item.get("macro", {}))

    # 5. 组装 Prompt (Prompt Engineering)
    prompt = (
        f"[User History]: {history_seq}\n"
        f"[Target Context]: {target_time}\n"
        f"[Intent Analysis]: {pref_text}\n"
        f"[Search categories]: {macro_hint}\n"
        f"Based on the above, next the user is most likely to visit:"
        # f"Refining this broad intent into a specific scene, the user is most likely to visit:"
    )

    return prompt