from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from utils import load_config

config = load_config()
model = CLIPModel.from_pretrained(config["clip_model_id"]) 
processor = CLIPProcessor.from_pretrained(config["clip_model_id"])

cos = torch.nn.CosineSimilarity()

def get_similar_images(image, file_name):
    retrieved_dir = f'./data/retrieved/{file_name}/images'
    if not os.path.exists(retrieved_dir):
        print(f"Warning: No retrieved images for Sample {file_name}")
        return
    
    image_preprocessed = processor(images=image, return_tensors="pt")['pixel_values']
    image_features = model.get_image_features(image_preprocessed)
    
    list_of_folders = os.listdir(f'./data/retrieved/{file_name}/images')
    for folder in list_of_folders:
        list_of_images = os.listdir(f'./data/retrieved/{file_name}/images/{folder}')
        images2 = []
        for img in list_of_images:
            try:
                img2 = Image.open(f'./data/retrieved/{file_name}/images/{folder}/{img}')
                images2.append(img2)
            except:
                continue
        if len(images2) == 0:
            continue
        images2_preprocessed = processor(images=images2, return_tensors="pt")['pixel_values']
        images2_features = model.get_image_features(images2_preprocessed)
        similarity = cos(image_features, images2_features)
        values, indices = similarity.topk(min(3, len(images2)))
        if not os.path.exists(f'./data/filtered/{file_name}/image_data'):
            os.makedirs(f'./data/filtered/{file_name}/image_data')
        for index in indices:
            extention = list_of_images[index].split('.')[-1]
            if images2[index].mode in ['RGBA', 'P']:
                images2[index] = images2[index].convert('RGB')
            images2[index].save(f'./data/filtered/{file_name}/image_data/{folder}_{index}.{extention}')
        