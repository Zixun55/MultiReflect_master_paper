import pandas as pd
import json
import os

def check_results():
    # 1. 讀取標準答案
    df_gt = pd.read_csv('./data/original/VERITE.csv')
    
    correct_count = 0
    analyzed_count = 0
    
    # 初始化各類別統計字典: { label: [總數, 答對數] }
    label_stats = {
        "true": [0, 0],
        "miscaptioned": [0, 0],
        "out-of-context": [0, 0]
    }

    print(f"{'Sample':<8} | {'Ground Truth':<15} | {'Model Prediction':<16} | {'Status'}")
    print("-" * 65)

    for idx, row in df_gt.iterrows():
        json_path_with_evi = f'./data/generated/{idx}/verification.json'
        json_path_no_evi = f'./data/generated/{idx}/verification_noevi.json'
        
        json_path = None
        if os.path.exists(json_path_with_evi):
            json_path = json_path_with_evi
        elif os.path.exists(json_path_no_evi):
            json_path = json_path_no_evi
            
        if json_path:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    response = data.get('response', '').upper()
                    
                    # 提取模型判斷
                    if 'TRUE' in response:
                        prediction = 'true'
                    elif 'FALSE' in response or 'MISCAPTIONED' in response:
                        prediction = 'miscaptioned'
                    elif 'OUT-OF-CONTEXT' in response:
                        prediction = 'out-of-context'
                    else:
                        prediction = 'unknown'
                    
                    # 取得原始標籤
                    ground_truth = str(row['label']).strip().lower()
                    
                    # 比對
                    is_correct = (prediction == ground_truth)
                    
                    # 更新統計數據
                    analyzed_count += 1
                    if is_correct:
                        correct_count += 1
                    
                    # 更新類別統計
                    if ground_truth in label_stats:
                        label_stats[ground_truth][0] += 1  # 該類別總數
                        if is_correct:
                            label_stats[ground_truth][1] += 1  # 該類別正確數
                    
                    status = "✅" if is_correct else "❌"
                    print(f"{idx:<8} | {ground_truth:<15} | {prediction:<16} | {status}")
                    
            except Exception as e:
                print(f"解析樣本 {idx} 失敗: {e}")
        
    # 輸出統計報告
    if analyzed_count > 0:
        print("\n" + "="*65)
        print(f"原架構類別統計 (Original Baseline Breakdown)")
        print("-" * 65)
        print(f"{'Label Type':<15} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
        print("-" * 65)
        
        for label, counts in label_stats.items():
            t_count = counts[0]
            c_count = counts[1]
            acc = (c_count / t_count * 100) if t_count > 0 else 0
            print(f"{label:<15} | {t_count:<8} | {c_count:<8} | {acc:.2f}%")
            
        accuracy = correct_count / analyzed_count
        print("-" * 65)
        print(f"總分析樣本數: {analyzed_count}")
        print(f"正確預測數:   {correct_count}")
        print(f"總體準確率 (Accuracy): {accuracy:.2%}")
        print("="*65)
    else:
        print("找不到任何已生成的 verification.json 檔案。")

# 執行
check_results()