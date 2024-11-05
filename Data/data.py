import pandas as pd
import os

# 文件路径
val_path = 'val.parquet'
test_path = 'test.parquet'

# 读取 parquet 文件
val_df = pd.read_parquet(val_path)
test_df = pd.read_parquet(test_path)

# 定义输出 JSON 文件的路径
val_json_path = os.path.splitext(val_path)[0] + '.json'
test_json_path = os.path.splitext(test_path)[0] + '.json'

# 将 DataFrame 转换为 JSON 格式并保存
val_df.to_json(val_json_path, orient='records', lines=True)
test_df.to_json(test_json_path, orient='records', lines=True)

print(f"文件已成功转换并保存：\n{val_json_path}\n{test_json_path}")
