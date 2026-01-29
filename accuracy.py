import pandas as pd
import json
import os

def check_results():
    df_gt = pd.read_csv('./data/original/VERITE.csv')
    
    correct_count = 0
    analyzed_count = 0
    details = []

    print(f"{'Sample':<8} | {'Ground Truth':<15} | {'Model Prediction':<10} | {'Status'}")
    print("-" * 60)

    for idx, row in df_gt.head(50).iterrows():
    # for idx, row in df_gt.iterrows():
        json_path = f'./data/generated/{idx}/verification.json'
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    response = data.get('response', '').upper()
                    
                    # 提取模型判斷
                    if 'TRUE' in response:
                        prediction = 'true'
                    elif 'FALSE' in response:
                        prediction = 'miscaptioned'
                    elif 'OUT-OF-CONTEXT' in response:
                        prediction = 'out-of-context'
                    else:
                        prediction = 'unknown'
                    
                    # 取得原始標籤
                    ground_truth = str(row['label']).strip().lower()
                    
                    # 比對
                    is_correct = (prediction == ground_truth)
                    if is_correct:
                        correct_count += 1
                    
                    analyzed_count += 1
                    status = "✅" if is_correct else "❌"
                    
                    print(f"{idx:<8} | {ground_truth:<15} | {prediction:<10} | {status}")
                    
            except Exception as e:
                print(f"解析樣本 {idx} 失敗: {e}")
        
    if analyzed_count > 0:
        accuracy = correct_count / analyzed_count
        print("-" * 60)
        print(f"總分析樣本數: {analyzed_count}")
        print(f"正確預測數: {correct_count}")
        print(f"總體準確率 (Accuracy): {accuracy:.2%}")
    else:
        print("找不到任何已生成的 verification.json 檔案。")

# 執行
check_results()