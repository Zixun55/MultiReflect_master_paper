import json
import os
from PIL import Image
from utils import get_llava_cot_response

IMAGE_PLACEHOLDER = '<image>'

def verification_prompt(image_path, caption, text_evidences, image_evidences):
    images = []
    image_orig = Image.open(image_path)
    images.append(image_orig)
    
    prompt_str = (
        f"{IMAGE_PLACEHOLDER} (Input Image)\n"
        "### Task: High-Precision Fact-Checker ###\n"
        "You are an expert fact-checker. You must strictly verify if the Caption matches the Evidence.\n\n"
        
        "### Mandatory Step-by-Step Analysis ###\n"
        "1. **Extraction**: Identify specific claims from the Caption: [Event], [Location], [Date], [Key Figures].\n"
        "2. **Evidence Comparison**: Check each claim against the provided Text and Image Evidences.\n"
        "3. **Conflict Detection**: If ANY detail (especially the Year or City) in the Evidence differs from the Caption, the result is NOT True.\n\n"
        
        "### Final Verdict Rules ###\n"
        "- **[FALSE]**: Output this if the Evidence DIRECTLY CONTRADICTS the Caption's details (e.g., different year, different city, or different people).\n"
        "- **[OUT-OF-CONTEXT]**: Output this if the image is authentic but was clearly taken during a different historical event than the caption claims.\n"
        "- **[TRUE]**: Output this ONLY if the Evidence fully supports the specific claims of the Caption without any discrepancies.\n\n"
        
        "### Formatting Requirement ###\n"
        "Start your response with: 'Verdict: [TRUE/FALSE/OUT-OF-CONTEXT]'. Then provide your step-by-step reasoning table.\n\n"
        f"Caption to verify: {caption}\n\n"
        "Evidences:\n"
    )
    
    for i, text in enumerate(text_evidences):
        prompt_str += f"Text Evidence {i+1}: {text}\n"
        
    prompt_str += "Image Evidences:\n"
    for i, img_ev_path in enumerate(image_evidences):
        prompt_str += f"Image Evidence {i+1}: {IMAGE_PLACEHOLDER}\n"
        images.append(Image.open(img_ev_path))
        
    return {
        "messages" : [{
            "role": "user",
            "content": prompt_str
        }],
        "images" : images
    }

def get_response_subs(file_name, image_path, caption, text_evidences, image_evidences, client):
    prompt = verification_prompt(image_path, caption, text_evidences, image_evidences)
    response = get_llava_cot_response(prompt,client)
    if not os.path.exists(f"./data/generated/{file_name}/"):
        os.makedirs(f"./data/generated/{file_name}/")
    with open(f"./data/generated/{file_name}/verification.json", "w") as f:
        json.dump({"response": response, "num text evidence": len(text_evidences), "num image evidence": len(image_evidences)}, f)