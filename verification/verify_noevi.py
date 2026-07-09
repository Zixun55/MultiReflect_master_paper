import json
import os
from PIL import Image
from utils import get_llava_cot_response

IMAGE_PLACEHOLDER = '<image>'

def verification_prompt(image_path, caption, consistency_score):
    image = Image.open(image_path)
    
    instruction = (
        f"{IMAGE_PLACEHOLDER}\n"
        "### [URGENT] CRITICAL VERIFICATION TASK ###\n"
        "Our internal consistency filter (CLIP+BLIP) has FLAGGED this pair as [INCONSISTENT].\n"
        f"FILTER CONFIDENCE SCORE: {consistency_score:.4f}\n\n"
        
        "### Your Mission ###\n"
        "The caption claims this image shows a specific event, but our filters suspect it is [Out-of-Context].\n"
        "You must act as a 'Fact-Checking Detective'. Do not be fooled by general similarities.\n"
        "Look for tiny details (text, clothing, weather, shadows, background entities) that prove this image DOES NOT belong to this caption.\n\n"
        
        f"Caption: {caption}\n\n"
        "### Analysis Requirement ###\n"
        "1. Identify any visual contradictions. "
        "Look for specific details (text, clothing, technology, environment) that seem mismatched. "
        "If the image looks consistent, state 'No major contradictions found'.\n"
        
        "2. Explain why the consistency filter might have assigned a score of {consistency_score:.4f}.\n\n"
        
        "--- FINAL OUTPUT FORMAT ---\n"
        "Final Verdict: [TRUE, FALSE, or OUT-OF-CONTEXT]\n"
        "Confidence: [Score 0-1]"
    )
    
    return { 
        "messages" : [{
            "role": "user",
            "content": instruction
        }],
        "images" : [image]
    }

def verification_prompt_when_no_consistency(image_path, caption):
    image = Image.open(image_path)
    
    instruction = (
        f"{IMAGE_PLACEHOLDER}\n"
        "### Task: Skeptical Visual Analyst ###\n"
        "You are verifying a claim WITHOUT external evidence. You must find internal contradictions between the Image and the Caption.\n\n"
        
        "### Decision Logic ###\n"
        "1. **[FALSE]**: Output this if the image CONTRADICTS the caption's core logic. "
        "(e.g., Caption says 'Heavy rain causes flood' but the image shows dry ground and sunny sky; "
        "or Caption says 'Protest in 2024' but people are using technology from 2005).\n"
        
        "2. **[OUT-OF-CONTEXT]**: Output this if the image looks real and matches the 'type' of event, "
        "but you suspect it belongs to a different time or place (e.g., generic pandemic photos used for a specific new outbreak).\n"
        
        "3. **[TRUE]**: Output this ONLY if the visual details (signs, weather, clothing, technology) "
        "perfectly align with the specific event described.\n\n"
        
        f"Caption: {caption}\n\n"
        "### Analysis Requirement ###\n"
        "First, describe 3 visual details that either support or contradict the caption. "
        "Then, provide your Verdict: [Verdict] and Confidence: [Score]."
    )
    
    return { 
        "messages" : [{
            "role": "user",
            "content": instruction
        }],
        "images" : [image]
    }

def get_response_subs(file_name, image_path, caption, consistency_score, client):
    if consistency_score:
        prompt = verification_prompt(image_path, caption, consistency_score)
    else:
        prompt = verification_prompt_when_no_consistency(image_path, caption)

    response = get_llava_cot_response(prompt,client)
    if not os.path.exists(f"./data/generated/{file_name}/"):
        os.makedirs(f"./data/generated/{file_name}/")
    with open(f"./data/generated/{file_name}/verification_noevi.json", "w") as f:
        json.dump({"response": response}, f)