import argparse
import os
import re
import json
import torch
import string
import time
from tqdm import tqdm
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2VLProcessor
import pandas as pd
from qwen_vl_utils import process_vision_info

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def call_llm(prompt: str, generator, max_new_tokens: int = 128) -> str:
    messages = [{"role": "user", "content": prompt}]
    output = generator(
                messages,
                max_new_tokens=max_new_tokens,
                top_p=None,
                do_sample=False)
    return output[0]["generated_text"][-1]['content'].strip()


def agent_response(query: str, visual_facts: str, document: str, score: float, generator, history: str = ""):
    core_instruction = (
        "1. SUBSTANTIAL CONTRADICTION ONLY: Only label as [MC] if there is a 'Hard Mismatch' in Core Facts "
        "(e.g., Wrong Year, Wrong City, or Wrong Main Event). \n"
        "2. TOLERANCE FOR SECONDARY DETAILS: Do not label as [MC] for minor descriptive differences "
        "(e.g., different adjectives, lighting, or background objects) if the main event matches.\n"
        "3. PRESUMPTION OF TRUTH: If the Visual Facts and the Document both generally support the "
        "Caption's main claim, you MUST label it as [True].\n"
        "4. SELECTIVE SKEPTICISM: Be aggressive against clear lies (2018 vs 2024), but be supportive if the evidence is just incomplete rather than contradictory.\n"
        "5. ENTITY ALIGNMENT: Check if the 'Visual Facts' and 'Document' confirm the exact Year and Location mentioned in the Caption. If either source gives a different Year/Place, you MUST defend an [MC] verdict during the debate."
    )
    
    if history:
        prompt = f"""You are a professional fact-checking agent in a multi-agent debate.
            Your current evidence reliability score is {score}/6.0.

            [FACT-CHECKING CONTEXT]
            - Original Caption: {query}
            - Visual Facts (from the image): {visual_facts}
            - Your Assigned Document: {document}

            Other agents' responses:
            {history}

            TASK: Based on the visual facts, your document, and the points raised by other agents:
            - If your score is higher, point out flaws in others' reasoning.
            - If others have stronger evidence (higher scores) showing a factual mismatch, re-evaluate your stance.
            {core_instruction}
            Provide your answer (True or MC) and reasoning.
            Format: 'Answer: {{}}. Explanation: {{}}.'"""
    else:
        prompt = f"""You are a professional fact-checking agent. 
            Your evidence reliability score is {score}/6.0.

            [FACT-CHECKING CONTEXT]
            - Original Caption: {query}
            - Visual Facts (from the image): {visual_facts}
            - Your Assigned Document: {document}

            TASK: Provide your initial independent verdict by comparing the Caption with the Visual Facts and your Document.
            {core_instruction}

            Format: 'Answer: {{}}. Explanation: {{}}.'"""

    return call_llm(prompt, generator)

def aggregate_responses(query: str, visual_facts: str, responses: List[str], scores: List[float], generator):
    joined = "\n".join([f"Agent {i+1} (Reliability Score: {scores[i]}): {r}" for i, r in enumerate(responses)])
    prompt = f"""You are a master aggregator (Judge) resolving conflicts in a multi-modal fact-checking debate.

                [REFERENCE DATA]
                - Original Caption: {query}
                - Visual Facts (What is actually in the photo): {visual_facts}
                Agent Debates with Reliability Scores:
                {joined}

                Instructions:
                Instructions:
                1. DETECTING MISMATCH: If ANY agent points out a SPECIFIC mismatch between the 'Visual Facts' (e.g., a sign, a face, a year) and the 'Caption', you MUST treat this as highly suspicious.
                2. CORE FACT CHECK: A mismatch in Year, Location, or Key Person is an automatic [MC].
                3. QUALITY OF EVIDENCE: Do not blindly follow the majority. Even a lower-score agent might find a 'Smoking Gun' (concrete evidence of a lie). If one agent provides a specific date or location from their document that contradicts the caption, prioritize [MC].
                4. TRUE VERDICT: Use [True] ONLY if NO significant factual contradictions were raised by any agent.
                5. YOU MUST CONCLUDE YOUR RESPONSE WITH THE FOLLOWING FORMAT:
                Final Verdict: <Label>
                Reason: <Brief Explanation>

                Replace <Label> with exactly one of: [True or MC]."""
    
    return call_llm(prompt, generator)


def multi_agent_debate(query: str, visual_facts: str, documents: List[str], scores: List[float], generator, num_rounds: int = 3):
    records = {}
    num_agents = len(documents)
    agent_outputs = []

    # Round 1
    records["round1"] = {"answers": [], "explanations": []}
    for i in range(num_agents):
        response = agent_response(query, visual_facts, documents[i], scores[i], generator)
        # 解析 Answer 與 Explanation
        ans_start = response.find("Answer: ") + len("Answer: ")
        exp_start = response.find("Explanation: ") + len("Explanation: ")
        answer = response[ans_start:response.find("Explanation")].strip()
        explanation = response[exp_start:].strip()
        
        records["round1"]["answers"].append(answer)
        records["round1"]["explanations"].append(explanation)
        agent_outputs.append(response)
    
    records["round1"]["aggregation"] = aggregate_responses(query, visual_facts, agent_outputs, scores, generator)
    
    # Additional rounds
    final_aggregation = records["round1"]["aggregation"]
    for t in range(1, num_rounds):
        round_key = f"round{t+1}"
        records[round_key] = {"answers": [], "explanations": []}
        new_outputs = []
        
        for i in range(num_agents):
            history = "\n".join([f"Agent {j+1}: {agent_outputs[j]}" for j in range(num_agents) if j != i])
            
            response = agent_response(query, visual_facts, documents[i], scores[i], generator, history)
            
            ans_start = response.find("Answer: ") + len("Answer: ")
            exp_start = response.find("Explanation: ") + len("Explanation: ")
            answer = response[ans_start:response.find("Explanation")].strip()
            explanation = response[exp_start:].strip()
            
            records[round_key]["answers"].append(answer)
            records[round_key]["explanations"].append(explanation)
            new_outputs.append(response)
            
        agent_outputs = new_outputs
        
        pred_ans_list = [normalize_answer(ans) for ans in records[round_key]["answers"]]
        prev_pred_ans_list = [normalize_answer(ans) for ans in records[f"round{t}"]["answers"]]
        
        flag = True
        for k in range(len(pred_ans_list)):
            if pred_ans_list[k] in prev_pred_ans_list[k] or prev_pred_ans_list[k] in pred_ans_list[k]:
                continue
            else:
                flag = False
                break
        
        if flag:
            final_aggregation = records[f"round{t}"]["aggregation"]
            break
        else:
            records[round_key]["aggregation"] = aggregate_responses(query, visual_facts, agent_outputs, scores, generator)
            final_aggregation = records[round_key]["aggregation"]

    records["final_aggregation"] = final_aggregation
    return records

def get_visual_facts(image_path, model, processor, device="cuda"):
    prompt = (
        "Identify and list specific facts in this image for fact-checking: "
        "1. Textual signs, dates, or names. "
        "2. Identifiable landmarks or buildings. "
        "3. People's clothing, uniforms, or specific items. "
        "4. Environmental cues (weather, season). "
        "Just state the facts you SEE. Do not guess."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    args = parser.parse_args()

    verite_path = os.path.join(args.data_path, "./original/VERITE.csv")
    if not os.path.exists(verite_path):
        print(f"Error: {verite_path} not found.")
        return
    df_verite = pd.read_csv(verite_path)

    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

    set_seed(42)
    print(f"Loading model: {MODEL_NAME}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        quantization_config=bnb_config, 
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        trust_remote_code=True, 
        device_map="auto"
    )

    processor = Qwen2VLProcessor.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)

    ranking_dir = os.path.join(args.data_path, "ranking_score")
    if not os.path.exists(ranking_dir):
        print(f"Error: Directory {ranking_dir} does not exist.")
        return

    # 找出所有是數字的資料夾名稱，並按數字大小排序
    folder_indices = sorted([f for f in os.listdir(ranking_dir) if f.isdigit()], key=int)
    print(f"Detected {len(folder_indices)} cases. Starting batch MADAM-RAG...")

    # 執行批次辯論
    output_filename = f"debating/final_madam_results_rounds{args.num_rounds}.jsonl"
    output_path = os.path.join(args.data_path, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_results = []
    for idx_str in tqdm(folder_indices, desc="Processing Cases"):
        idx = int(idx_str)
        csv_path = os.path.join(ranking_dir, idx_str, "text_data", "final_scores.csv")
        
        if not os.path.exists(csv_path):
            continue
        
        try:
            original_caption = df_verite.iloc[idx]['caption']
            rel_image_path = df_verite.iloc[idx]['image_path']
            image_path = os.path.abspath(os.path.join(args.data_path, "original", rel_image_path))

            print(f"Extrating visual facts for Sample {idx}...")
            visual_facts = get_visual_facts(image_path, model, processor)

            # score.csv
            df = pd.read_csv(csv_path)
            top_k = df.sort_values(by='total', ascending=False).head(3)
            
            # query_with_caption = (
            #     f"Original Caption: '{original_caption}'\n\n"
            #     "Task: Verify if this specific caption is [True] or [Miscaptioned (MC)] based on the visual evidence and provided documents."
            # )

            result = multi_agent_debate(
                query=original_caption, 
                visual_facts=visual_facts,
                documents=top_k['evidence'].tolist(), 
                scores=top_k['total'].tolist(), 
                generator=generator, 
                num_rounds=args.num_rounds
            )
            
            result["folder_idx"] = idx
            all_results.append(result)

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
        except Exception as e:
            print(f"Error processing folder {idx}: {e}")
            continue
        finally:
            torch.cuda.empty_cache()

    print(f"\nAll tasks finished. Total cases processed: {len(all_results)}")
    print(f"Results saved to: {output_path}")
    end_time = time.time()
    total_duration = end_time - start_time

    print("\n" + "="*30)
    print(f"總執行時間: {total_duration:.2f} 秒")
        
    minutes = int(total_duration // 60)
    seconds = total_duration % 60
    print(f"格式化時間: {minutes} 分 {seconds:.2f} 秒")
    print("="*30)

if __name__ == "__main__":
    main()