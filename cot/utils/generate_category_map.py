import pandas as pd
import json
from transformers import pipeline
import torch
from tqdm import tqdm

# --- 1. 配置 ---
# 用于分类的零样本模型
MODEL_NAME = "cross-encoder/nli-roberta-base"

# 所有可能的宏观类别 (与您其他脚本保持一致)
MACRO_CATEGORIES = [
    "food & dining",
    "arts & entertainment",
    "shopping & retail",
    "health & wellness",
    "travel & transportation",
    "professional & public services",
    "outdoors & nature"
]

# --- 2. 主函数 ---

def create_fine_to_macro_map(dataset):
    # 包含所有不重复细分类别的CSV文件
    categories_csv_file = f"../datasets/{dataset}/dataIntegration/all_category_counts.csv"

    # 输出的JSON映射文件名
    output_csv_file = f"../datasets/{dataset}/dataIntegration/categories_with_macro.csv"

    try:
        df = pd.read_csv(categories_csv_file)
        category_column_name = df.columns[0]
        fine_categories = df[category_column_name].dropna().unique().tolist()
        print(f"成功从 '{categories_csv_file}' 加载了 {len(fine_categories)} 个不重复的细分类别。")
    except FileNotFoundError:
        print(f"错误：找不到类别文件 '{categories_csv_file}'。请确保文件存在。")
        return

    # --- b. 初始化零样本分类器 ---
    print(f"正在加载零样本分类模型: '{MODEL_NAME}'...")
    try:
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=device
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # --- c. 使用批处理进行高效分类 ---
    premise_template = "This is a place of type: {}"
    contextualized_categories = [premise_template.format(cat) for cat in fine_categories]

    print("开始对所有细分类别进行宏观类别预测...")
    # 模板可以帮助模型更好地理解上下文，特别是对于短语
    hypothesis_template = "The category for this location is {}."

    # pipeline在处理列表时会自动显示进度条
    results = classifier(
        contextualized_categories,
        MACRO_CATEGORIES,
        hypothesis_template=hypothesis_template,
        multi_label=False,  # 我们只需要Top-1的预测
        batch_size=64  # 根据您的硬件调整
    )
    print("所有类别预测完成。")

    # --- d. 构建并保存映射表 ---
    fine_to_macro_map = {}
    prefix = "This is a place of type: "
    prefix_len = len(prefix)

    for result in results:
        # 从 "This is a place of type: Bar" 中提取出 "Bar"
        full_premise = result['sequence']
        original_fine_category = full_premise[prefix_len:]

        macro_category = result['labels'][0]
        fine_to_macro_map[original_fine_category] = macro_category

    # --- e. 【核心修改】将映射结果添加为DataFrame的新列 ---
    print("正在将预测结果添加为新列 'macro_category'...")
    # 使用 pandas 的 .map() 函数，高效地应用映射
    df['macro_category'] = df[category_column_name].map(fine_to_macro_map)

    # --- f. 保存带有新列的CSV文件 ---
    print(f"正在将结果保存到 '{output_csv_file}'...")
    df.to_csv(output_csv_file, index=False, encoding='utf-8')

    print("带有宏观类别的新CSV文件生成成功！")

    # --- g. 打印一些示例以供快速检查 ---
    print("\n--- 结果预览 ---")
    print(df.head(10).to_string())


# --- 主执行模块 ---
if __name__ == "__main__":
    create_fine_to_macro_map('nyc')
