import pandas as pd
import json
import os
import re

def evaluate_fusion_system():
    # 1. 讀取標準答案
    gt_path = './data/original/VERITE.csv'
    if not os.path.exists(gt_path):
        print(f"❌ 找不到標準答案檔: {gt_path}")
        return
    df_gt = pd.read_csv(gt_path)
    
    # 2. 載入 MADAM-RAG 辯論結果
    madam_jsonl = './data/debating/final_madam_results_rounds3.jsonl'
    madam_preds = {}
    label_map = {
        "TRUE": "true", "MC": "miscaptioned", "FALSE": "miscaptioned",
        "MISCAPTIONED": "miscaptioned", "OOC": "out-of-context", "OUT-OF-CONTEXT": "out-of-context"
    }

    if os.path.exists(madam_jsonl):
        with open(madam_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    idx = str(data["folder_idx"]).strip()
                    final_text = data.get("final_aggregation", "")
                    match = re.search(r"Final Verdict:\s*\*?\[?(True|False|MC|OOC|Miscaptioned|Out-of-context)\]?\.?", final_text, re.IGNORECASE)
                    if match:
                        raw_pred = match.group(1).upper()
                        madam_preds[idx] = label_map.get(raw_pred, "unknown")
                except: continue

    # 3. 初始化統計變數
    correct_count = 0
    total_count = 0
    # 建立一個結構：{ 'label_name': [總數, 答對數] }
    label_stats = {
        "true": [0, 0],
        "miscaptioned": [0, 0],
        "out-of-context": [0, 0]
    }
    
    print(f"{'Sample':<8} | {'Ground Truth':<15} | {'Model Prediction':<16} | {'Status'}")
    print("-" * 65)

    for idx_int, row in df_gt.iterrows():
        idx_str = str(idx_int).strip()
        
        base_exists = os.path.exists(f'./data/generated/{idx_int}/verification.json') or \
                  os.path.exists(f'./data/generated/{idx_int}/verification_noevi.json')
        
        if base_exists:
            ground_truth = str(row['label']).strip().lower()
            prediction = None
            source = None

            if idx_str in madam_preds:
                prediction = madam_preds[idx_str]
                source = "MADAM-RAG"
            else:
                path_with_evi = f'./data/generated/{idx_int}/verification.json'
                path_no_evi = f'./data/generated/{idx_int}/verification_noevi.json'

                json_path = None
                if os.path.exists(path_with_evi):
                    json_path = path_with_evi
                    source = "Original_with_Evi"
                elif os.path.exists(path_no_evi):
                    json_path = path_no_evi
                    source = "Original_no_Evi"

                if json_path:
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            res = data.get('response', '').upper()
                            if 'TRUE' in res: prediction = 'true'
                            elif 'FALSE' in res or 'MISCAPTIONED' in res: prediction = 'miscaptioned'
                            elif 'OUT-OF-CONTEXT' in res: prediction = 'out-of-context'
                            source = "Original"
                    except: pass

            if source:
                total_count += 1
                is_correct = (prediction == ground_truth)
                
                # 更新類別統計
                if ground_truth in label_stats:
                    label_stats[ground_truth][0] += 1 # 總數 +1
                    if is_correct:
                        label_stats[ground_truth][1] += 1 # 答對數 +1

                if is_correct:
                    correct_count += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{idx_str:<8} | {ground_truth:<15} | {prediction:<16} | {status}")

    # 4. 產出最終統計報告
    print("\n" + "="*65)
    print(f"各類別標籤統計 (Label Breakdown)")
    print("-" * 65)
    print(f"{'Label Type':<15} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 65)
    
    for label, counts in label_stats.items():
        total_l = counts[0]
        correct_l = counts[1]
        acc_l = (correct_l / total_l * 100) if total_l > 0 else 0
        print(f"{label:<15} | {total_l:<8} | {correct_l:<8} | {acc_l:.2f}%")
    
    print("-" * 65)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    print(f"總有效樣本數: {total_count}")
    print(f"總體正確數:   {correct_count}")
    print(f"總體準確率:   {accuracy:.22f}%")
    print("="*65)

if __name__ == "__main__":
    evaluate_fusion_system()