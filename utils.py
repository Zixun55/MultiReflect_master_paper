import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import os
import yaml
from transformers import CLIPProcessor, CLIPModel

def load_config(config_file='config.yaml'):
    if not os.path.exists(config_file):
        return {"model_id": "Qwen/Qwen2-VL-2B-Instruct"}
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ImageTextToImageModel():
    def __init__(self, checkpoint):
        print(f"正在載入 Qwen2-VL 模型: {checkpoint}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype="auto",
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint)

def get_llava_cot_response(prompt, client):
    torch.cuda.empty_cache()
    
    # 1. 取得原始問題與圖片
    messages = prompt["messages"]
    raw_images = prompt["images"]
    
    # 提取文字內容
    if isinstance(messages[0]["content"], list):
        question_text = next((item['text'] for item in messages[0]['content'] if item['type'] == 'text'), "")
    else:
        question_text = messages[0]["content"]

    # 2. 構建 Qwen2-VL 特有的輸入格式
    content = []
    for img in raw_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": question_text})
    
    qwen_messages = [{"role": "user", "content": content}]

    # 3. 處理輸入張量
    text_prompt = client.processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(qwen_messages)
    
    inputs = client.processor(
        text=[text_prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    max_pixels=320*320,
    min_pixels=128*128
    ).to(client.model.device)

    # 4. 生成答案
    with torch.no_grad():
        generated_ids = client.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = client.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    torch.cuda.empty_cache()
    return response

class CLIPConsistency:
    def __init__(self, model_id="openai/clip-vit-large-patch14-336"):
        print(f"正在載入 CLIP-336 模型: {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = CLIPModel.from_pretrained(
            model_id, 
            use_safetensors=True 
        ).to(self.device)
        
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def get_similarity(self, image_path, caption):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[caption], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            similarity = (image_embeds * text_embeds).sum(dim=-1).item()
        return similarity
    
# JSON 輔助函式保持不變
def load_json(file_path):
    if not os.path.exists(file_path): return {}
    with open(file_path, "r") as file: return json.load(file)
def save_json(data, file_path):
    with open(file_path, "w") as file: json.dump(data, file, indent=4)
def append_jsonl(data, file_path):
    with open(file_path, "a") as file: file.write(json.dumps(data) + "\n")