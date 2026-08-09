from consistency import clip_consistency
from consistency import blip_consistency
from eval_check import llava_evalcheck
from retrieval import text_retrieval, image_retrieval
from filtering import filtering_text, filtering_image
from ranking import combined
from verification import verify
from verification import verify_noevi
from utils import *

# from openai import OpenAI
import pandas as pd
import os
import re
import time
from PIL import Image

# api_key = os.getenv("OPENAI_API_KEY")

# client = OpenAI(api_key=api_key, timeout=100)


config = load_config()
client = ImageTextToImageModel(config['model_id'])

def consistency_response(image_path, caption, idx):
    # CLIP-336做第一次consistency
    response = clip_consistency.get_response(image_path, caption, client)
    verdict = 0

    if not os.path.exists(f"{config['output_path']}/generated/{idx}/"):
        os.makedirs(f"{config['output_path']}/generated/{idx}/")
    save_json({"consistency_response": response}, f"{config['output_path']}/generated/{idx}/consistency_response.json")

    if "<verdict>TRUE</verdict>" in response:
        verdict = 1
    elif "<verdict>FALSE</verdict>" in response:
        verdict = 0
    else:
        if 'TRUE' in response:
            verdict = 1
        elif 'FALSE' in response:
            verdict = 0 
        else:
            if 'true' in response.lower():
                verdict = 1
            elif 'false' in response.lower():
                verdict = 0 
            else:
                verdict = 0
    
    score = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", response)
    
    if len(score) == 0:
        score = 0
    else:
        score = float(score[0])

    # BLIP-ITM做第二次consistency
    if verdict == 0 and score >= 0.25:
        print(f"Sample {idx}: CLIP Passed ({score}). Running BLIP verification...")
        
        blip_v, blip_s = blip_consistency.blip_verdict(image_path, caption)
                
        save_json({
            "clip_response": response,
            "blip_score": blip_s,
            "blip_verdict": blip_v
        }, f"{config['output_path']}/generated/{idx}/consistency_response.json")

        if blip_v == 1:
            verdict = 1

    return verdict, score

# def consistency_response(image_path, caption, idx):
#     # 建立輸出路徑 (維持原有名稱與邏輯)
#     if not os.path.exists(f"{config['output_path']}/generated/{idx}/"):
#         os.makedirs(f"{config['output_path']}/generated/{idx}/")

#     # 只執行 BLIP-ITM 做 consistency
#     # 這裡 blip_v 對應判定結果 (1 or 0)，blip_s 對應匹配分數 (score)
#     blip_v, blip_s = blip_consistency.blip_verdict(image_path, caption)
    
#     # 為了對應回傳變數名稱，將其指派給 verdict 與 score
#     verdict = blip_v
#     score = blip_s
                
#     # 儲存 JSON (維持原有的儲存習慣，僅保留 BLIP 相關資訊)
#     save_json({
#         "blip_score": score,
#         "blip_verdict": verdict
#     }, f"{config['output_path']}/generated/{idx}/consistency_response.json")

#     # 回傳變數名稱保持為 verdict, score
#     return verdict, score

def eval_check_response(image_path, caption, text_evidences, image_evidences, idx, first):
    if first:
        response = llava_evalcheck.get_response_first(image_path, caption, client)
        if not os.path.exists(f"{config['output_path']}/generated/{idx}/"):
            os.makedirs(f"{config['output_path']}/generated/{idx}/")
        append_jsonl({"eval_check_response": response, "check": "first"}, f"{config['output_path']}/generated/{idx}/eval_check_response.jsonl")
        response_lines = response.split("\n")
        needs_retrieval = True
        if 'Yes' in response_lines[0] or 'yes' in response_lines[0].lower():
            needs_retrieval = True
        elif 'No' in response_lines[0] or 'no' in response_lines[0].lower():
            needs_retrieval = False
        else:
            if 'Yes' in response or 'yes' in response.lower():
                needs_retrieval = True
            elif 'No' in response or 'no' in response.lower():
                needs_retrieval = False
            else:
                needs_retrieval = True
        return needs_retrieval
    else:
        response = llava_evalcheck.get_response_subs(image_path, caption, text_evidences, image_evidences, client)
        if not os.path.exists(f"{config['output_path']}/generated/{idx}/"):
            os.makedirs(f"{config['output_path']}/generated/{idx}/")
        append_jsonl({"eval_check_response": response, "check": "subsequent"}, f"{config['output_path']}/generated/{idx}/eval_check_response.json")
        response_lines = response.split("\n")
        needs_retrieval = True
        if '[Continue to Use Evidence]' in response_lines[0]:
            needs_retrieval = False
        elif '[No Retrieval]' in response_lines[0]:
            needs_retrieval = False
        elif '[Retrieval]' in response_lines[0]:
            needs_retrieval = True
        else:
            if '[Continue to Use Evidence]' in response or '[No Retrieval]' in response:
                needs_retrieval = False
            elif '[Retrieval]' in response:
                needs_retrieval = True
            else:
                needs_retrieval = True
        return needs_retrieval

def get_evidences(image_path, caption, idx):
    text_path = f"{config['output_path']}/retrieved/{idx}/text_data"
    img_path = f"{config['output_path']}/retrieved/{idx}/image_data"
    text_done = os.path.exists(f"{text_path}/ddg_search.json")
    img_done = os.path.exists(f"{img_path}/ddg_data.json")

    if not text_done:
        print(f"Sample {idx}: 執行文字檢索...")
        with open(image_path, "rb") as f:
            image_content = f.read()

        # Text Retrieval
        res = text_retrieval.get_data(caption, image_content)
        wikipedia_search, google_search, bing_search, inverse_google_search, \
        inverse_bing_search, inverse_google_data, inverse_bing_data, ddg_search = res

        text_path = f"{config['output_path']}/retrieved/{idx}/text_data"
        if not os.path.exists(text_path):
            os.makedirs(text_path)

        save_json(wikipedia_search, f"{text_path}/wikipedia_search.json")
        save_json(google_search, f"{text_path}/google_search.json")
        save_json(bing_search, f"{text_path}/bing_search.json")
        save_json(inverse_google_search, f"{text_path}/inverse_google_search.json")
        save_json(inverse_bing_search, f"{text_path}/inverse_bing_search.json")
        save_json(inverse_google_data if isinstance(inverse_google_data, list) else [], f"{text_path}/inverse_google_data.json")
        save_json(inverse_bing_data if isinstance(inverse_bing_data, list) else [], f"{text_path}/inverse_bing_data.json")
        save_json(ddg_search, f"{text_path}/ddg_search.json")
    else:
        print(f"Sample {idx}: 文字檢索已存在。")

    if not img_done:
        # Image Retrieval
        print(f"Sample {idx}: 執行圖片檢索...")
        google_image_data, bing_image_data, commons_data, ddg_data = image_retrieval.get_image_data(caption, idx)
        
        img_path = f"{config['output_path']}/retrieved/{idx}/image_data"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
    
        save_json(google_image_data, f"{img_path}/google_image_data.json")
        save_json(bing_image_data, f"{img_path}/bing_image_data.json")
        save_json(commons_data, f"{img_path}/commons_data.json")
        save_json(ddg_data, f"{img_path}/ddg_data.json")
    else:
        print(f"Sample {idx}: 圖片檢索已存在。")

def init_pipeline(image_path, caption, idx):
    print('Checking Consistency for Sample', idx)
    try:
        consistency_verdict, consistency_score = consistency_response(image_path, caption, idx)
        print("Consistency Verdict:", consistency_verdict)
        print("Consistency Score:", consistency_score)
        print('-'*50)
    except:
        consistency_verdict = 0
        print("Consistency Error for", idx)
        print('-'*50)
    if consistency_verdict == 1:
        # Move to is retrieval needed
        print('Checking if Retrieval is Needed for Sample', idx)
        try:
            extra_evidence_need = eval_check_response(image_path, caption, [], [], idx, first=True)
            print("First Eval Check:", extra_evidence_need)
            print('-'*50)
        except:
            extra_evidence_need = True
            print("First Eval Check Error for", idx)
            print('-'*50)
        if extra_evidence_need:
            # return
            # Move to Retrieval
            print('Retrieval Needed for Sample', idx)
            get_evidences(image_path, caption, idx)
            print('Retrieval Done for Sample', idx)
            print('-'*50)
            # Move to Filtering
            print('Filtering Text for Sample', idx)
            filtering_text.get_all_text_filtered(idx, caption)
            print('Text Filtering Done for Sample', idx)
            print('-'*50)
            print('Filtering Image for Sample', idx)
            img = Image.open(image_path)
            filtering_image.get_similar_images(img, idx)
            print('Image Filtering Done for Sample', idx)
            print('-'*50)
            # Move to Ranking
            print('Ranking Text and Image for Sample', idx)
            combined.get_text_scores(idx, caption, image_path, client)
            combined.get_image_scores(idx, caption, image_path, client)
            print('Ranking Done for Sample', idx)
            print('-'*50)
            # Check for each evidence
            print('Checking for each Evidence for Sample', idx)
            text_evidences = pd.read_csv(f"{config['output_path']}/ranking_score/{idx}/text_data/final_scores.csv")
            text_evidences = text_evidences[text_evidences["total"]>0]
            if len(text_evidences) == 0:
                text_evidences = []
            else:
                text_evidences = text_evidences.sort_values(by="total", ascending=False)["evidence"].tolist()
            image_evidences = pd.read_csv(f"{config['output_path']}/ranking_score/{idx}/image_data/final_scores.csv")
            image_evidences = image_evidences[image_evidences["total"]>0]
            if len(image_evidences) == 0:
                image_evidences = []
            else:
                image_evidences = image_evidences.sort_values(by="total", ascending=False)["evidence"].tolist()
            selected_text_evidences = []
            selected_image_evidences = []
            curr_text_idx = 0
            curr_image_idx = 0

            max_text_limit = 5 
            max_image_limit = 2

            while True:
                print(f"--- 迴圈嘗試中: 目前文字證據數 {len(selected_text_evidences)} / 圖片數 {len(selected_image_evidences)} ---")
                if len(selected_text_evidences) >= max_text_limit:
                    break
                can_add_text = curr_text_idx < len(text_evidences)
                can_add_image = curr_image_idx < len(image_evidences) and len(selected_image_evidences) < max_image_limit
                
                if not can_add_text and not can_add_image:
                    break
                if curr_text_idx == len(text_evidences) and curr_image_idx == len(image_evidences):
                    break
                if curr_text_idx < len(text_evidences):
                    selected_text_evidences.append(text_evidences[curr_text_idx])
                    curr_text_idx += 1
                if curr_image_idx < len(image_evidences) and len(selected_image_evidences) < max_image_limit:
                    selected_image_evidences.append(f"{config['output_path']}/filtered/{idx}/image_data/"+image_evidences[curr_image_idx])
                    curr_image_idx += 1
                if not eval_check_response(image_path, caption, selected_text_evidences, selected_image_evidences, idx, first=False):
                    break
            print('Checking for each Evidence Done for Sample', idx)
            print('-'*50)
            # Move to Verification
            print('Verifying for Sample', idx)
            verify.get_response_subs(idx, image_path, caption, selected_text_evidences, selected_image_evidences, client)
            print('Verification Done for Sample', idx)
            print('-'*50)
            print('-'*50)
        else:
            print('No Retrieval Needed for Sample', idx)
            print('-'*50)
            print('Verifying for Sample', idx)
            verify_noevi.get_response_subs(idx, image_path, caption, None, client)
            print('Verification Done for Sample', idx)
            print('-'*50)
            print('-'*50)
    else:
        print('Consistency Verdict is False for Sample', idx)
        print('-'*50)
        print('Verifying for Sample', idx)
        verify_noevi.get_response_subs(idx, image_path, caption, consistency_score, client)
        print('Verification Done for Sample', idx)
        print('-'*50)
        print('-'*50)
        
start_time = time.time()
df = pd.read_csv(f"{config['data_path']}/VERITE.csv")

input_num = len(df)
# input_num = 100

for idx in range(input_num):
    try:
        caption = df.iloc[idx]['caption']
        image_path = f"{config['data_path']}/{df.iloc[idx]['image_path']}"
        if not os.path.exists(image_path):
            print(f"跳過 Sample {idx}: 找不到圖片檔案 {image_path}")
            continue
        init_pipeline(image_path, caption, idx)

        torch.cuda.empty_cache()
    except Exception as e:
        print(e, idx)
        continue

end_time = time.time()
total_duration = end_time - start_time

print("\n" + "="*30)
print(f"總執行樣本數: {input_num}")
print(f"總執行時間: {total_duration:.2f} 秒")
    
minutes = int(total_duration // 60)
seconds = total_duration % 60
print(f"格式化時間: {minutes} 分 {seconds:.2f} 秒")
print("="*30)

# caption = df.iloc[7]['caption']
# image_path = f"{config['data_path']}/{df.iloc[7]['image_path']}"
# init_pipeline(image_path, caption, 7)

# torch.cuda.empty_cache()