import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Salesforce/blip-itm-base-coco"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForImageTextRetrieval.from_pretrained(
    model_id, 
    use_safetensors=True,
    ignore_mismatched_sizes=True,
    local_files_only=False
).to(device)

def get_blip_score(image_path, caption):
    raw_image = Image.open(image_path).convert("RGB")
    
    inputs = processor(images=raw_image, text=caption, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # ITM 輸出匹配(index 1)與不匹配(index 0)
        outputs = model(**inputs)
        # 使用 Softmax 轉為機率
        itm_scores = torch.nn.functional.softmax(outputs.itm_score, dim=1)
        itm_score = itm_scores[:, 1].item() # 取得 "Matched" 的機率
        
    return itm_score

def blip_verdict(image_path, caption, threshold=0.2):
    score = get_blip_score(image_path, caption)
    print("BLIP Score:", score)
    # 如果分數大於門檻值，判定為 True (Consistent)
    verdict = 1 if score > threshold else 0
    return verdict, score