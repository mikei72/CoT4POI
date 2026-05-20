import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Correct macro-category labels by matching category names.")
parser.add_argument("--target_file", default="../dataIntegration/categories_with_macro.csv")
parser.add_argument("--reference_file", required=True)
parser.add_argument("--output_file", default="../dataIntegration/corrected_categories_with_macro.csv")
args = parser.parse_args()

df_target = pd.read_csv(args.target_file)
df_ref = pd.read_csv(args.reference_file)

print(f"待校正数据行数: {len(df_target)}")
print(f"参考表数据行数: {len(df_ref)}")
print("-" * 50)

mapping_dict = dict(zip(df_ref['CategoryName'], df_ref['macro_category']))

target_categories = set(df_target['CategoryName'])
ref_categories = set(mapping_dict.keys())

missing_categories = target_categories - ref_categories

print(f"本次共有 {len(missing_categories)} 个分类在参考表中缺失 (无法校正):")
if len(missing_categories) > 0:
    uncorrected_rows = df_target[df_target['CategoryName'].isin(missing_categories)]
    print(uncorrected_rows[['CategoryName', 'Count', 'macro_category']])
else:
    print("完美！所有分类都在参考表中找到了对应项。")

print("-" * 50)

mapped_series = df_target['CategoryName'].map(mapping_dict)

df_target['macro_category'] = mapped_series.combine_first(df_target['macro_category'])

df_target.to_csv(args.output_file, index=False)
print(f"校正完成！结果已保存至: {args.output_file}")
print("前5行预览:")
print(df_target.head())
