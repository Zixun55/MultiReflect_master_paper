from sentence_transformers import SentenceTransformer, util
import os
import requests
import json
from bs4 import BeautifulSoup
import spacy

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36 Edg/122.0.0.0', 'Referer': 'https://www.google.com/'}
nlp = spacy.load('en_core_web_sm')

def compute_similarity(text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores

def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def get_top_k_snippets_wiki(file_name, caption):
    snippets = []
    with open(f'./data/retrieved/{file_name}/text_data/wikipedia_search.json', 'r') as f:
        data = json.load(f)
    if "query" in data.keys():
        if "search" in data["query"].keys():
            for search in data["query"]["search"]:
                try:
                    snippets.append(remove_html_tags(search["snippet"]))
                except:
                    continue
    if snippets == []:
        return
    cosine_scores = compute_similarity(caption, snippets)
    # top_k = min(3, len(snippets))
    values, indices = cosine_scores.topk(len(snippets))
    indices = indices.tolist()[0]
    cleaned_text = {}
    done = 0
    for index in indices:
        if done == 3:
            break
        try:
            url = "https://en.wikipedia.org/w/api.php?action=parse&page=" + data["query"]["search"][index]["title"] + "&format=json"
            response = requests.get(url)
            page = json.loads(response.text)
            cleaned_text[index] = {"pagetext": remove_html_tags(page["parse"]["text"]["*"]), "title": data["query"]["search"][index]["title"], "timestamp": data["query"]["search"][index]["timestamp"]}
            done += 1
        except:
            continue
    if not os.path.exists(f'./data/filtered/{file_name}/text_data'):
        os.makedirs(f'./data/filtered/{file_name}/text_data')
    with open(f'./data/filtered/{file_name}/text_data/wiki.json', 'w') as f:
        json.dump(cleaned_text, f)

def get_top_k_snippets_ddg(file_name, caption):
    snippets = []
    input_path = f'./data/retrieved/{file_name}/text_data/ddg_search.json'
    
    if not os.path.exists(input_path):
        return

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list) or len(data) == 0:
        return

    for item in data:
        try:
            content = item.get("snippet") or item.get("body") or ""
            if content:
                snippets.append(remove_html_tags(content))
        except:
            continue

    if not snippets:
        return

    cosine_scores = compute_similarity(caption, snippets)
    values, indices = cosine_scores.topk(len(snippets))
    indices = indices.tolist()[0]
    
    cleaned_text = {}
    done = 0
    for index in indices:
        if done == 3: break
        try:
            link = data[index].get("link") or data[index].get("href")
            page = requests.get(link, headers=headers, timeout=10)
            soup = BeautifulSoup(page.content, 'html.parser')
            text = soup.get_text()
            
            cleaned_text[index] = {
                "pagetext": text, 
                "title": data[index].get("title"), 
                "timestamp": None, 
                "link": link
            }
            done += 1
        except:
            continue

    output_dir = f'./data/filtered/{file_name}/text_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f'{output_dir}/ddg.json', 'w') as f:
        json.dump(cleaned_text, f)
        
def get_timestamp_metadata_google(obj):
    timestamp_tags = ['article:published_time', "datepublished", "date", "pubdate", "release_date", "cdc:first_published", "dc.date", "dc.date.issued"]
    for tag in timestamp_tags:
        if tag in obj.keys():
            return obj[tag]
    return None

def get_top_k_snippets_google(file_name, caption):
    snippets = []
    with open(f'./data/retrieved/{file_name}/text_data/google_search.json', 'r') as f:
        data = json.load(f)
    if "items" in data.keys():
        for item in data["items"]:
            try:
                snippets.append(remove_html_tags(item["snippet"]))
            except:
                continue
    if snippets == []:
        return
    cosine_scores = compute_similarity(caption, snippets)
    # top_k = min(3, len(snippets))
    values, indices = cosine_scores.topk(len(snippets))
    indices = indices.tolist()[0]
    cleaned_text = {}
    done = 0
    for index in indices:
        if done == 3:
            break
        try:
            link = data["items"][index]["link"]
            page = requests.get(link, headers=headers)
            soup = BeautifulSoup(page.content, 'html.parser')
            text = soup.get_text()
            timestamp=None
            if "pagemap" in data["items"][index].keys():
                if "metatags" in data["items"][index]["pagemap"].keys():
                    timestamp = get_timestamp_metadata_google(data["items"][index]["pagemap"]["metatags"][0])
            cleaned_text[index] = {"pagetext": text, "title": data["items"][index]["title"], "timestamp": timestamp, "link": link}
            done += 1
        except:
            continue
    if not os.path.exists(f'./data/filtered/{file_name}/text_data'):
        os.makedirs(f'./data/filtered/{file_name}/text_data')
    with open(f'./data/filtered/{file_name}/text_data/google.json', 'w') as f:
        json.dump(cleaned_text, f)
        
def get_top_k_snippets_bing(file_name, caption):
    snippets = []
    with open(f'./data/retrieved/{file_name}/text_data/bing_search.json', 'r') as f:
        data = json.load(f)
    if "webPages" in data.keys():
        if "value" in data["webPages"].keys():
            for item in data["webPages"]["value"]:
                try:
                    snippets.append(remove_html_tags(item["snippet"]))
                except:
                    continue
    if snippets == []:
        return
    cosine_scores = compute_similarity(caption, snippets)
    # top_k = min(3, len(snippets))
    values, indices = cosine_scores.topk(len(snippets))
    indices = indices.tolist()[0]
    cleaned_text = {}
    done = 0
    for index in indices:
        if done == 3:
            break
        try:
            link = data["webPages"]["value"][index]["url"]
            page = requests.get(link, headers=headers)
            soup = BeautifulSoup(page.content, 'html.parser')
            text = soup.get_text()
            cleaned_text[index] = {"pagetext":text, "link": link, "title": data["webPages"]["value"][index]["name"], "timestamp": None}
            done += 1
        except:
            continue
    if not os.path.exists(f'./data/filtered/{file_name}/text_data'):
        os.makedirs(f'./data/filtered/{file_name}/text_data')
    with open(f'./data/filtered/{file_name}/text_data/bing.json', 'w') as f:
        json.dump(cleaned_text, f)
        
def get_inverse_google(file_name):
    input_path = f'./data/retrieved/{file_name}/text_data/inverse_google_data.json'
    
    if not os.path.exists(input_path):
        return

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return

        new_data = {}
        for idx, item in enumerate(data):
            pagetext = item.get('text', '')
            link = item.get('url', '')
            if pagetext:
                new_data[idx] = {"pagetext": pagetext, "link": link}
        
        output_dir = f'./data/filtered/{file_name}/text_data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(f'{output_dir}/inverse_google.json', 'w') as f:
            json.dump(new_data, f)
            
    except Exception as e:
        print(f"Error in filtering_text.get_inverse_google for {file_name}: {e}")
    
def get_clean_data_new_inverse_bing(file_name):
    with open(f'./data/retrieved/{file_name}/text_data/inverse_bing_data.json', 'r') as f:
        data = json.load(f)
    new_data = {}
    for idx, item in enumerate(data):
        new_data[idx] = {"pagetext": item['text'], "link": item["url"]}
    if not os.path.exists(f'./data/filtered/{file_name}/text_data'):
        os.makedirs(f'./data/filtered/{file_name}/text_data')
    with open(f'./data/filtered/{file_name}/text_data/inverse_bing.json', 'w') as f:
        json.dump(new_data, f)
        
import langdetect

def split_text_into_paragraphs(text, max_words=250):
    # Split the text into paragraphs or chucks of max_words which ever is smaller
    if langdetect.detect(text) != 'en':
        return None
    paragraphs = []
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    paragraph = ""
    for sentence in sentences:
        if len(paragraph.split()) + len(sentence.split()) <= max_words:
            paragraph += sentence
        else:
            paragraph = paragraph.strip()
            paragraphs.append(paragraph)
            paragraph = sentence
    if paragraph:
        paragraphs.append(paragraph)
    return paragraphs        

def get_most_similar_paragraphs(file_name, caption):
    filtered_dir = f'./data/filtered/{file_name}/text_data'
    if not os.path.exists(filtered_dir): return {}

    files = os.listdir(f'./data/filtered/{file_name}/text_data')
    selected_paragraphs = {}
    for file in files:
        if file == 'paragraphs.json': continue
        try:
            with open(f'./data/filtered/{file_name}/text_data/{file}', 'r') as f:
                data = json.load(f)
            print(file)
            if not data: continue
            
            paragraphs = []
            paragraphs_meta = []
            for idx, temp in data.items():
                text = temp.get("pagetext", "")
                if not text or len(text.strip()) < 10: continue
                try:
                    paras = split_text_into_paragraphs(text)
                    paragraph_meta = []
                    if paras is None:
                        continue
                    for i in range(len(paras)):
                        paragraph_meta.append({"text": paras[i], "title": temp.get("title"), "timestamp": temp.get("timestamp"), "link": temp.get("link")})
                    paragraphs += paras
                    paragraphs_meta += paragraph_meta
                except:
                    continue

            if not paragraphs:
                continue

            embeddings = model.encode(paragraphs, convert_to_tensor=True)
            query_embedding = model.encode(caption, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)
            top_k = min(3, len(paragraphs))
            values, indices = cosine_scores.topk(top_k)
            indices = indices.tolist()[0]
            temp = []
            for index in indices:
                temp.append(paragraphs_meta[index])
            selected_paragraphs[file] = temp
        except:
            print(f"Error in {file_name} {file}")
            continue
    return selected_paragraphs

def get_all_text_filtered(file_name, caption):
    get_top_k_snippets_wiki(file_name, caption)
    get_top_k_snippets_google(file_name, caption)
    get_top_k_snippets_bing(file_name, caption)
    get_top_k_snippets_ddg(file_name, caption)
    get_inverse_google(file_name)
    get_clean_data_new_inverse_bing(file_name)
    if not os.path.exists(f'./data/filtered/{file_name}/text_data'):
        os.makedirs(f'./data/filtered/{file_name}/text_data')
    with open(f'./data/filtered/{file_name}/text_data/paragraphs.json', 'w') as f:
        json.dump(get_most_similar_paragraphs(file_name, caption), f)