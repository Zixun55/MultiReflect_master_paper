import io
import spacy
import requests
import json
import os
from googleapiclient.discovery import build
from utils import load_config
from ddgs import DDGS
import random
import time
import google.generativeai as genai
from PIL import Image
from io import BytesIO

config = load_config()
nlp = spacy.load('en_core_web_sm')
headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36 Edg/122.0.0.0', 'Referer': 'https://www.google.com/'}

def get_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

def create_query(entities, sep=" "):
    query = ""
    for ent in entities:
        query += ent[0] + sep
    return query

def get_google_images(caption):
    entities = get_entities(caption)
    base_query = create_query(entities) if len(entities) > 0 else caption
    query = f"!gi {base_query}"
    
    results = []
    for attempt in range(3):
        try:
            wait_time = random.uniform(2, 5) 
            time.sleep(wait_time)
            
            with DDGS() as ddgs:
                ddgs_gen = ddgs.images(query, max_results=5)
                results = list(ddgs_gen)
                if results:
                    return results
        except Exception as e:
            print(f"第 {attempt+1} 次嘗試失敗: {e}")
            time.sleep(5)
            continue
            
    return results
def get_commons_response(caption):
    entities = get_entities(caption)
    responses = {}
    url = "https://commons.wikimedia.org/w/api.php"
    
    search_list = [ent[0] for ent in entities] if entities else [caption]

    for query in search_list:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": f"File:{query}",
            "srlimit": 3
        }
        
        try:
            r = requests.get(url, params=params, timeout=30, headers={'User-Agent': 'ThesisFactCheck/1.0'})
            data = r.json()
            
            if "query" in data and "search" in data["query"]:
                responses[query] = {
                    "pages": [
                        {"title": item["title"], "id": item["pageid"]} 
                        for item in data["query"]["search"]
                    ]
                }
            print(f"Commons 搜尋成功: {query}")
        except Exception as e:
            print(f"Commons 搜尋失敗 ({query}): {e}")
            
    return responses
    
# def get_bing_responses(caption):
#     entities = get_entities(caption)
#     if len(entities) > 0:
#         query = create_query(entities, "+")
#     else:
#         query = entities
#     url = "https://api.bing.microsoft.com/v7.0/images/search"
#     headers = {
#         "Content-Type": "multipart/form-data",
#         "Ocp-Apim-Subscription-Key": config["bing_ocp_apim_subscription_key"], 
#     }
#     try:
#         response = requests.get(url, headers=headers, params={"q": query, "count": 10}, timeout=60)
#         data = response.json()
#         return data
#     except Exception as e:
#         return {}
    
def download_google_images(data, file_name):
    if not data:
        return
        
    save_dir = f'./data/retrieved/{file_name}/images/google_images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, item in enumerate(data):
        try:
            response = requests.get(item['image'], headers=headers, timeout=30)
            if response.status_code == 200:
                ext = item['image'].split('.')[-1].split('?')[0].lower()
                if ext not in ['jpg', 'jpeg', 'png', 'webp']:
                    ext = 'jpg'
                
                with open(f'{save_dir}/{idx+1}.{ext}', 'wb') as f:
                    f.write(response.content)
        except Exception:
            continue

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')
def download_commons_images(data, file_name):
    save_dir = f'./data/retrieved/{file_name}/images/commons_images'
    
    for query, value in data.items():
        if 'pages' in value:
            for page in value['pages']:
                title = page['title']
                if not title.startswith('File:'):
                    title = f"File:{title}"

                if not title.lower().endswith(VALID_EXTENSIONS):
                    print(f"跳過不支援的 Commons 格式: {title}")
                    continue

                info_url = "https://commons.wikimedia.org/w/api.php"
                params = {
                    "action": "query",
                    "format": "json",
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "titles": title
                }
                
                try:
                    res = requests.get(info_url, params=params, timeout=30, headers={'User-Agent': 'ThesisBot/1.0'})
                    info_data = res.json()
                    
                    pages = info_data.get("query", {}).get("pages", {})
                    for p_id, p_info in pages.items():
                        if "imageinfo" in p_info:
                            actual_download_url = p_info["imageinfo"][0]["url"]
                            
                            if not actual_download_url.lower().endswith(VALID_EXTENSIONS):
                                print(f"跳過非位圖 URL: {actual_download_url}")
                                continue

                            print(f"嘗試下載 Commons 圖片: {page['id']}")
                            img_res = requests.get(actual_download_url, timeout=30, headers={'User-Agent': 'ThesisBot/1.0'})
                            
                            if img_res.status_code == 200:
                                try:
                                    # 嘗試從記憶體中開啟圖片
                                    img_temp = Image.open(BytesIO(img_res.content))
                                    # 驗證圖片內容是否完整
                                    img_temp.verify()
                                    
                                    # 驗證通過，正式存檔
                                    os.makedirs(save_dir, exist_ok=True)
                                    extension = actual_download_url.split('.')[-1].lower()
                                    # 修正副檔名格式 (jpeg -> jpg)
                                    if extension == 'jpeg': extension = 'jpg'
                                    
                                    file_path = f"{save_dir}/{page['id']}.{extension}"
                                    with open(file_path, 'wb') as f:
                                        f.write(img_res.content)
                                    print(f"成功下載並驗證 Commons 圖片: {page['id']}")
                                
                                except (io.UnidentifiedImageError, ValueError) as pil_err:
                                    print(f"圖片格式無效或損壞，ID {page['id']}: {pil_err}")
                                    continue
                                
                except Exception as e:
                    print(f"下載 Commons 圖片失敗 ({page['id']}): {e}")
                    continue
                
# def download_bing_images(data, file_name):
#     if 'value' in data.keys():
#         for value in data['value']:
#             try:
#                 response = requests.get(value['contentUrl'], headers=headers, timeout=60)
#                 if not os.path.exists(f'./data/retrieved/{file_name}/images/bing_images'):
#                     os.makedirs(f'./data/retrieved/{file_name}/images/bing_images')
#                 if "encodingFormat" in value.keys():
#                     extension = value["encodingFormat"]
#                 else:
#                     extension = value['contentUrl'].split('.')[-1]
#                 with open(f'./data/retrieved/{file_name}/images/bing_images/{value["imageId"]}.{extension}', 'wb') as f:
#                     f.write(response.content)
#             except Exception as e:
#                 continue

def get_ddg_images(caption):
    entities = get_entities(caption)
    query = create_query(entities) if len(entities) > 0 else caption
    
    results = []
    try:
        with DDGS() as ddgs:
            ddgs_gen = ddgs.images(query, max_results=5)
            results = list(ddgs_gen)
            
        import time
        time.sleep(2)
        
    except Exception as e:
        print(f"DuckDuckGo 圖片搜尋失敗: {e}")
        
    return results

def download_ddg_images(data, file_name):
    if not data:
        return
        
    save_dir = f'./data/retrieved/{file_name}/images/ddg_images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, item in enumerate(data):
        try:
            response = requests.get(item['image'], headers=headers, timeout=30)
            if response.status_code == 200:
                ext_raw = item['image'].split('.')[-1].split('?')[0].lower()
                valid_exts = ['jpg', 'jpeg', 'png', 'webp']
                if ext_raw not in valid_exts:
                    print(f"跳過不支援的格式: {ext_raw}")
                    continue # 直接跳過這個循環，不下載這個檔案
                ext = ext_raw
                
                with open(f'{save_dir}/{idx+1}.{ext}', 'wb') as f:
                    f.write(response.content)
        except Exception:
            continue
    
def get_image_data(caption, idx):
    commons_data = {}
    ddg_data = []

    try:
        commons_data = get_commons_response(caption)
        download_commons_images(commons_data, idx)
    except Exception as e:
        print(f"Commons 下載失敗: {e}")

    try:
        ddg_data = get_ddg_images(caption)
        download_ddg_images(ddg_data, idx)
    except Exception as e:
        print(f"DDG 下載失敗: {e}")

    # try:
    #     google_image_data = get_google_images(caption)
    #     download_google_images(google_image_data, idx)
    # except Exception as e:
    #     print(f"Google 下載失敗: {e}")

    bing_image_data = {}
    google_image_data = {}
    
    return google_image_data, bing_image_data, commons_data, ddg_data
