import pandas as pd
import os
import io


def get_distribution(dataset):
    output_filename = f'../datasets/{dataset}/dataIntegration/all_category_counts.csv'

    if dataset == 'nyc':
        file_paths = [
            '../datasets/nyc/raw/NYC_train.csv',
            '../datasets/nyc/raw/NYC_val.csv',
            '../datasets/nyc/raw/NYC_test.csv'
        ]
        dfs = []
        target_column = 'PoiCategoryName'

        print("开始读取文件...")
        for file_path in file_paths:
            df_temp = pd.read_csv(file_path)
            dfs.append(df_temp)
            print(f"  [成功] 读取: {file_path} (行数: {len(df_temp)})")

        if len(dfs) > 0:
            df = pd.concat(dfs, ignore_index=True)

            print(f"\n所有文件合并完成，总行数: {len(df)}")
        else:
            df = None

    elif dataset == 'ca':
        file_path = '../datasets/ca/raw/dataset_gowalla_ca_ne.csv'
        target_column = 'PoiCategoryId'

        print("开始读取文件...")
        df = pd.read_csv(file_path)
        print(f"  [成功] 读取: {file_path} (行数: {len(df)})")

    elif dataset == 'tky':
        file_path = '../datasets/tky/raw/dataset_TSMC2014_TKY.txt'
        target_column = 'PoiCategoryId'

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print("[警告] 检测到编码异常，启动容错读取并清洗数据...")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            content = content.replace('Caf\ufffd', 'Cafe').replace('Café', 'Cafe')

        data_buffer = io.StringIO(content)

        print("开始读取文件...")
        df = pd.read_csv(data_buffer, sep='\t', header=None)
        df.columns = ['UserId', 'PoiId', 'CategoryHash', 'PoiCategoryId',
                      'Latitude', 'Longitude', 'TimezoneOffset', 'UTCTime']
        print(f"  [成功] 读取: {file_path} (行数: {len(df)})")

    else:
        print("\n invalid dataset name")
        return

    if len(df) > 0:
        category_counts = df[target_column].value_counts()
        category_counts_df = category_counts.reset_index()
        category_counts_df.columns = ['CategoryName', 'Count']

        category_counts_df.to_csv(output_filename, index=False)

        print(f"\n统计结果已成功保存至 '{output_filename}'")
        print("-" * 30)
        print("前 5 个热门分类:")
        print(category_counts_df.head())
    else:
        print("\n[失败] 没有读取到任何有效的数据文件。")


