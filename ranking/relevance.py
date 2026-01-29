import os
import json
from PIL import Image
from utils import get_llava_cot_response

IMAGE_PLACEHOLDER = '<image>'

def get_text_to_image_relevance(text_evidence, image_path):
    image = Image.open(image_path)

    instruction = (
        f"{IMAGE_PLACEHOLDER}\n"
        """
        You'll be provided with an image, along with evidence. Your job is to determine if the evidence is relevant to the 
        determine the factual correctness of the image, and provides useful information to complete the task described in the instruction. 
        If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].
        """
        f"Text Evidence: {text_evidence}"
    )
    
    return {
      "messages" : [
          {
        "role": "user",
        "content": instruction
      }
      ],
      "images" : [image]
    }
    
def get_text_to_image_relevance_sample(file_name, image_path, client):
    responses = {}
    evidence_file = json.load(open(f'./data/filtered/{file_name}/text_data/paragraphs.json'))
    for key in evidence_file.keys():
        res_list = []
        for evidence in evidence_file[key]:
          try:
            prompt = get_text_to_image_relevance(evidence['text'], image_path)
            response = get_llava_cot_response(prompt, client)
            res_list.append(response)
          except:
            res_list.append("Error")
            continue
        responses[key] = res_list
    return responses

def get_image_to_text_relevance(image_evidence_path, caption):
    image = Image.open(image_evidence_path)

    instruction = (
        f"{IMAGE_PLACEHOLDER}\n"
        """
        You'll be provided with a text, along with an image evidence. Your job is to determine if the evidence is relevant to the 
        determine the factual correctness of the text, and provides useful information to complete the task described in the instruction. 
        If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].
        """
        f"Text: {caption}"
    )
    
    return {
      "messages" : [
          {
        "role": "user",
        "content": instruction
      }
      ],
      "images" : [image]
    }
    
def get_image_to_text_relevance_sample(file_name, caption, client):
    responses = {}
    evidences = os.listdir(f'./data/filtered/{file_name}/image_data/')
    for evidence in evidences:
      try:
        prompt = get_image_to_text_relevance(f'./data/filtered/{file_name}/image_data/'+evidence, caption)
        response = get_llava_cot_response(prompt, client)
        responses[evidence] = response
      except:
        responses[evidence] = "Error"
        continue
    return responses