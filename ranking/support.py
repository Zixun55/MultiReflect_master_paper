import os
import json
from PIL import Image
from utils import get_llava_cot_response

IMAGE_PLACEHOLDER = '<image>'

def get_text_support(caption, image_path, text_evidence):
    image = Image.open(image_path)
    
    instruction = (
        f"{IMAGE_PLACEHOLDER} (Input Image)\n"
        """
        You will receive an input text, input image and text evidence towards determining the factuality of the input.
        Your task is to evaluate if the input is fully supported by the information provided in the evidence.
        Use the following entailment scale to generate a score:
        - [Fully supported] - All information in input is supported by the evidence, or extractions
        from the evidence.
        - [Partially supported] - The input is supported by the evidence to some extent, but there
        is major information in the input that is not discussed in the evidence. For example, if
        the input asks about two concepts and the evidence only discusses either of them, it should
        be considered a [Partially supported].
        - [No support / Contradictory] - The input completely ignores evidence, is unrelated to the
        evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the
        instruction.
        Make sure to not use any external information/knowledge to judge whether the input is true or not.
        Only check whether the input is supported by the evidence, and not whether the input follows the instructions or not.
        
        Output Entailment like [Fully supported], [Partially supported] or [No support / Contradictory]
        """
        f"Input Text: {caption}\n"
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
    
def get_text_support_sample(file_name, caption, image_path, client):
    responses = {}
    evidence_file = json.load(open(f'./data/filtered/{file_name}/text_data/paragraphs.json'))
    for key in evidence_file.keys():
        res_list = []
        for evidence in evidence_file[key]:
            try:
              prompt = get_text_support(caption, image_path, evidence['text'])
              response = get_llava_cot_response(prompt, client)
              res_list.append(response)
            except:
              res_list.append("Error text")
              continue
        responses[key] = res_list
    return responses

def get_image_support(image_evidence_path, caption, image_path):
    
    images = []
    
    # 處理輸入圖片
    image_main = Image.open(image_path)
    images.append(image_main)
    
    # 處理圖片證據
    image_evidence = Image.open(image_evidence_path)
    images.append(image_evidence)

    instruction = (
        f"{IMAGE_PLACEHOLDER} (Input Image)\n" # 第一張圖
        f"{IMAGE_PLACEHOLDER} (Image Evidence)\n" # 第二張圖
        """
        You will receive an input text, input image and image evidence towards determining the factuality of the input.
        Your task is to evaluate if the input is fully supported by the information provided in the evidence.
        Use the following entailment scale to generate a score:
        - [Fully supported] - All information in input is supported by the evidence, or extractions
        from the evidence.
        - [Partially supported] - The input is supported by the evidence to some extent, but there
        is major information in the input that is not discussed in the evidence. For example, if
        the input asks about two concepts and the evidence only discusses either of them, it should
        be considered a [Partially supported].
        - [No support / Contradictory] - The input completely ignores evidence, is unrelated to the
        evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the
        instruction.
        Make sure to not use any external information/knowledge to judge whether the input is true or not.
        Only check whether the input is supported by the evidence, and not whether the input follows the instructions or not.
        
        Output Entailment like [Fully supported], [Partially supported] or [No support / Contradictory]
        """
        f"Input Text: {caption}"
    )
    
    return {
      "messages" : [
          {
        "role": "user",
        "content": instruction 
      }
      ],
      "images" : images
    }
    
def get_image_support_sample(file_name, caption, image_path, client):
    responses = {}
    evidences = os.listdir(f'./data/filtered/{file_name}/image_data/')
    for evidence in evidences:
      try:
        prompt = get_image_support(f'./data/filtered/{file_name}/image_data/'+evidence, caption, image_path)
        response = get_llava_cot_response(prompt, client)
        responses[evidence] = response
      except:
        responses[evidence] = "Error image"
        continue
    return responses