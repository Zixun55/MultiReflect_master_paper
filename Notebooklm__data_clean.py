import pandas as pd
import json
import os
import io

CSV_FILE_PATH = './data/NotebookLM_data.csv'

def refill_to_original_format():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"找不到檔案: {CSV_FILE_PATH}")
        return

    df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8-sig')
    
    for idx, group in df.groupby('ID'):
        target_dir = f'./data/retrieved/{idx}/text_data'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        evidence_list = []
        for _, row in group.iterrows():
            evidence_list.append({
                "title": str(row['title']),
                "link": str(row['link']),
                "snippet": str(row['snippet']),
                "timestamp": str(row['timestamp']),
                "source": str(row['source'])
            })
        
        file_path = os.path.join(target_dir, "NotebookLM_search.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(evidence_list, f, ensure_ascii=False, indent=4)
        
        print(f"ID {idx}：成功回填 {len(evidence_list)} 筆證據至 {file_path}")

if __name__ == "__main__":
    refill_to_original_format()