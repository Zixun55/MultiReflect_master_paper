import pandas as pd
import json
import os
import re

def evaluate_debate_only():
    # 1. 讀取標準答案 (Ground Truth)
    gt_path = './data/original/VERITE.csv'
    if not os.path.exists(gt_path):
        print(f"❌ 找不到標準答案檔: {gt_path}")
        return
    df_gt = pd.read_csv(gt_path)
    # 將 GT 轉成 dictionary 方便快速查找：{ '0': 'true', '1': 'out-of-context' ... }
    gt_dict = {str(i): str(row['label']).strip().lower() for i, row in df_gt.iterrows()}
    
    # 2. 載入 MADAM-RAG 辯論結果
    madam_jsonl = './data/debating/final_madam_results_rounds3.jsonl'
    label_map = {
        "TRUE": "true", "MC": "miscaptioned", "FALSE": "miscaptioned",
        "MISCAPTIONED": "miscaptioned", "OOC": "out-of-context", "OUT-OF-CONTEXT": "out-of-context"
    }

    correct_count = 0
    total_count = 0
    label_stats = {
        "true": [0, 0],
        "miscaptioned": [0, 0],
        "out-of-context": [0, 0]
    }

    print(f"{'Sample':<8} | {'Ground Truth':<15} | {'Debate Prediction':<18} | {'Status'}")
    print("-" * 70)

    if not os.path.exists(madam_jsonl):
        print(f"❌ 找不到辯論結果檔: {madam_jsonl}")
        return

    with open(madam_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                idx_str = str(data["folder_idx"]).strip()
                final_text = data.get("final_aggregation", "")
                
                # 使用 Regex 提取 Final Verdict
                match = re.search(r"Final Verdict:\s*\*?\[?(True|False|MC|OOC|Miscaptioned|Out-of-context)\]?\.?", final_text, re.IGNORECASE)
                
                if match:
                    raw_pred = match.group(1).upper()
                    prediction = label_map.get(raw_pred, "unknown")
                    ground_truth = gt_dict.get(idx_str)

                    if ground_truth:
                        total_count += 1
                        is_correct = (prediction == ground_truth)
                        
                        # 更新統計
                        label_stats[ground_truth][0] += 1
                        if is_correct:
                            label_stats[ground_truth][1] += 1
                            correct_count += 1
                        
                        status = "✅" if is_correct else "❌"
                        print(f"{idx_str:<8} | {ground_truth:<15} | {prediction:<18} | {status}")
            except Exception as e:
                continue

    # 3. 產出統計報告
    print("\n" + "="*70)
    print(f"📊 MADAM-RAG 辯論結果統計 (Debate Only)")
    print("-" * 70)
    print(f"{'Label Type':<15} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 70)
    
    for label, counts in label_stats.items():
        total_l = counts[0]
        correct_l = counts[1]
        if total_l > 0:
            acc_l = (correct_l / total_l * 100)
            print(f"{label:<15} | {total_l:<8} | {correct_l:<8} | {acc_l:.2f}%")
    
    print("-" * 70)
    if total_count > 0:
        accuracy = (correct_count / total_count * 100)
        print(f"辯論樣本總數: {total_count}")
        print(f"辯論正確總數: {correct_count}")
        print(f"辯論總體準確率: {accuracy:.4f}%")
    else:
        print("未偵測到有效的辯論預測數據。")
    print("="*70)

if __name__ == "__main__":
    evaluate_debate_only()