import spacy
import requests
import json
import os
from googleapiclient.discovery import build
from utils import load_config
from ddgs import DDGS

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

# def get_google_images(caption):
#     entities = get_entities(caption)
#     if len(entities) == 0:
#         query = caption
#     else:
#         query = create_query(entities)
#     # url = f"https://www.googleapis.com/customsearch/v1?key={config[""]}&cx=b7f8c7d4cce094056&q=" + query + "&searchType=image"
#     try:
#         service = build("customsearch", "v1", developerKey = config["google_customsearch_key"]) #google_customsearch_key
#         res = service.cse().list(q=query,cx="b7f8c7d4cce094056",searchType="image").execute()
#         # response = requests.get(url)
#         # data = response.json()
#         return res
#     except Exception as e:
#         return {}

def get_commons_response(caption):
    entities = get_entities(caption)
    responses = {}
    try:
        if len(entities) > 0:
            for ent in entities:
                query = ent[0]
                url = f"https://api.wikimedia.org/core/v1/commons/search/title"
                params = {'q': query, 'limit': 10}
                try:
                    response = requests.get(url, headers={"User-Agent": f"Evidence ({config['wiki_user_agent']})", "Authorization": f"Bearer {config['wiki_authorization_bearer_key']}"}, params=params, timeout=60) 
                    data = response.json()
                    responses[query] = data
                except Exception as e:
                    continue
            return responses
        else:
            query = entities
            url = f"https://api.wikimedia.org/core/v1/commons/search/title"
            params = {'q': query, 'limit': 10}
            try:
                response = requests.get(url, headers={"User-Agent": f"Evidence ({config['wiki_user_agent']})", "Authorization": f"Bearer {config['wiki_authorization_bearer_key']}"}, params=params, timeout=60)
                data = response.json()
                responses[query] = data
            except Exception as e:
                pass
            return responses
    except Exception as e:
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
    
# def download_google_images(data, file_name):
#     if 'items' in data.keys():
#         for idx, item in enumerate(data['items']):
#             try:
#                 response = requests.get(item['link'], timeout=60)
#                 if not os.path.exists(f'./data/retrieved/{file_name}/images/google_images'):
#                     os.makedirs(f'./data/retrieved/{file_name}/images/google_images')
#                 extension = ''
#                 if 'jpg' in item['link'].split('.')[-1]:
#                     extension = 'jpg'
#                 elif 'png' in item['link'].split('.')[-1]:
#                     extension = 'png'
#                 elif 'jpeg' in item['link'].split('.')[-1]:
#                     extension = 'jpeg'
#                 else:
#                     continue
#                 with open(f'./data/retrieved/{file_name}/images/google_images/{idx+1}.{extension}', 'wb') as f:
#                     f.write(response.content)
#             except Exception as e:
#                 continue

def download_commons_images(data, file_name):
    for value in data.values():
        if 'pages' in value.keys():
            for page in value['pages']:
                url = f"https://api.wikimedia.org/core/v1/commons/file/{page['title']}"
                try:
                    response = requests.get(url, headers={"User-Agent": f"Evidence ({config['wiki_user_agent']})", "Authorization": f"Bearer {config['wiki_authorization_bearer_key']}"}, timeout=60)
                    response = response.json()
                    if 'preferred' in response.keys():
                        new_url = response['preferred']['url']
                        new_response = requests.get(url, headers={"User-Agent": f"Evidence ({config['wiki_user_agent']})", "Authorization": f"Bearer {config['wiki_authorization_bearer_key']}"}, timeout=60)
                        if not os.path.exists(f'./data/retrieved/{file_name}/images/commons_images'):
                            os.makedirs(f'./data/retrieved/{file_name}/images/commons_images')
                        extension = new_url.split('.')[-1]
                        with open(f'./data/retrieved/{file_name}/images/commons_images/{page["id"]}.{extension}', 'wb') as f:
                            f.write(new_response.content)
                except Exception as e:
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
                ext = item['image'].split('.')[-1].split('?')[0]
                if ext.lower() not in ['jpg', 'jpeg', 'png']:
                    ext = 'jpg'
                
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

    google_image_data = {}
    bing_image_data = {}
    
    return google_image_data, bing_image_data, commons_data, ddg_data
