from PIL import Image
from utils import get_llava_cot_response

IMAGE_PLACEHOLDER = '<image>'

def get_prompt_first(image_path, caption):
    image = Image.open(image_path)
    
    instruction = (
        f"{IMAGE_PLACEHOLDER}\n"
        "Identify if this image-caption pair could be 'Miscaptioned' (wrong date/event) "
        "or 'Out-of-context'. Unless it is a 100% obvious common sense fact, "
        "you MUST answer [Yes] to seek web evidence. Answer [Yes] or [No].\n"
        f"Caption: {caption}"
    )
    
    return {
        "messages": [{
            "role": "user",
            "content": instruction
        }],
        "images": [image]
    }

def get_prompt_subs(image_path, caption, text_evidences, image_evidences):
    images = []
    
    image_main = Image.open(image_path)
    images.append(image_main)
    
    prompt_parts = [
        f"{IMAGE_PLACEHOLDER} (Original Image)\n",
        """
        Given a image and caption along with some external documents (evidences). 
        Your task is to determine whether the factuality of the image and caption can be fully
        verified by the evidence or if it requires further external verification.
        There are three cases:
        - If image and caption can be verified solely with the evidences, then respond with [Continue
        to Use Evidence].
        - If the sentence doesn't require any factual verification (e.g., a subjective sentence or a
        sentence about common sense), then respond with [No Retrieval].
        - If additional information is needed to verify, respond with [Retrieval].
        Please provide explanations for your judgments
        """,
        f"Caption: {caption}",
        "--- Evidences ---",
    ]
    
    # 處理文字證據
    for i in range(len(text_evidences)):
        prompt_parts.append(f"Text Evidence {i+1}: {text_evidences[i]}")

    # 處理圖片證據
    prompt_parts.append("Image Evidences:")
    for i in range(len(image_evidences)):
        # 在文本中插入圖片標記，用於標示證據圖片的位置
        prompt_parts.append(f"Image Evidence {i+1}: {IMAGE_PLACEHOLDER}")
        
        # 讀取證據圖片物件並添加到 images 列表中
        image_ev = Image.open(image_evidences[i])
        images.append(image_ev) 

    full_instruction = "\n".join(prompt_parts)
    
    return {
        "messages" : [{
            "role": "user",
            "content": full_instruction
        }],
        "images" : images
    }

def get_response_first(image_path, caption, client):
    prompt = get_prompt_first(image_path, caption)
    return get_llava_cot_response(prompt,client)

def get_response_subs(image_path, caption, text_evidences, image_evidences, client):
    prompt = get_prompt_subs(image_path, caption, text_evidences, image_evidences)
    return get_llava_cot_response(prompt,client)