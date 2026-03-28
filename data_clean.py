import pandas as pd
import os
from utils import load_config

# 1. 載入配置
config = load_config()
csv_path = f"{config['data_path']}/VERITE.csv"
output_path = f"{config['data_path']}/VERITE_CLEANED.csv"

# 2. 讀取原始資料
df = pd.read_csv(csv_path)
print(f"原始資料筆數: {len(df)}")

# 3. 定義檢查函數
def check_image_exists(row):
    # 根據你的 main.py 邏輯組合路徑
    full_path = f"{config['data_path']}/{row['image_path']}"
    return os.path.exists(full_path)

# 4. 執行篩選
# apply 會檢查每一列，回傳 True/False，最後只保留 True 的資料
df_cleaned = df[df.apply(check_image_exists, axis=1)].copy()

# 5. 輸出結果與存檔
print(f"清理後剩餘筆數: {len(df_cleaned)}")
print(f"共刪除了 {len(df) - len(df_cleaned)} 筆找不到圖片的資料。")

df_cleaned.to_csv(output_path, index=False)
print(f"✅ 清理完成！新資料集已存至: {output_path}")