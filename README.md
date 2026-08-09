# 執行步驟
1. 執行環境
```bash
conda activate llava_env
```
2. 執行main
```bash
python main.py
```
3. 執行衝突辯論
```bash
python run_madam_rag.py --data_path ./data
```
4. 計算準確率
```bash
python3 verification/evaluate_final_results.py
```

# 資料集
[VERITE dataset](https://github.com/stevejpapad/image-text-verification)

# data 資料夾
```
data/
├── debating              # Evidence Debating 結果
├── filtered              # Evidence Filtering 結果
├── generated             # Consistency Checking 及未辯論過的最終結果
├── original              # 資料集
├── ranking_score         # Evidence Ranking 結果
└── retrieved             # 檢索資料
```
**Notice:** 
1. 執行 main.py 前需將 debating、filtered、generated、ranking_score、retrieved 資料夾刪除，新的資料不會自動覆蓋掉。
2. 若沒有要重新檢索，不需將 retrieved 資料夾刪除，已有檢索資料就不會再次執行檢索。

# 目錄結構
```
data/
├── main.py                            # 主程式
├── consistency                        # Consistency Checking
│   ├── blip_consistency.py            # BLIP 的 Consistency Checking
│   └── clip_consistency.py            # CLIP 的 Consistency Checking
├── eval_check                         # Evidence Checking
│   └── llava_evalcheck.py
├── filtering                          # Evidence Filtering
│   ├── filtering_image.py             # 圖片過濾
│   └── filtering_text.py              # 文字過濾
├── ranking                            # Evidence Ranking
│   ├── annotated_sources_final.csv    # 網域來源資料
│   ├── authoritative.py
│   ├── combined.py                    # 計算總分
│   ├── freshness.py
│   ├── relevance.py
│   ├── support.py
│   └── useful.py
├── retrieval # Retrieval
│   ├── image_retrieval.py             # 圖片檢索
│   └── text_retrieval.py              # 文字檢索
├── verification
│   ├── evaluate_final_results.py      # 計算最終準確率
│   ├── verify_noevi.py                # 免辯論、免檢索的結果
│   └── verify.py                      # 免辯論的結果
├── accuracy.py                        # 不含辯論的準確率計算
└── run_madam_rag.py                   # Evidence Debating
```