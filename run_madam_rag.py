import argparse
import os
import re
import json
import torch
import string
from tqdm import tqdm
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from transformers import Qwen2VLForConditionalGeneration
import pandas as pd

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


def agent_response(query: str, document: str, score: float, generator, history: str = ""):
    core_instruction = (
        "- CRITICAL: Compare dates (years), specific locations, and named entities in the Document against the Question.\n"
        "- If the Document confirms the image but mentions a DIFFERENT YEAR or LOCATION than the caption, you MUST label it as 'MC' (Miscaptioned).\n"
        "- Be skeptical. Do not ignore small factual contradictions just because the overall theme matches."
    )
    # 根據分數給予不同指令
    if history:
        prompt = f"""You are a professional fact-checking agent. Your evidence has a MultiReflect Reliability Score of {score}/6.0.

            Question: {query}
            Your Document: {document}

            Other agents' responses:
            {history}

            Task: Based on your score and the provided evidence, resolve any conflicts with other agents. 
            {core_instruction}
            - If your score is higher, defend your point and challenge others' evidence.
            - If others have higher scores, critically re-evaluate your stance.
            Provide your answer (True, OOC, or MC) and reasoning.
            Format: 'Answer: {{}}. Explanation: {{}}.'"""
    else:
        prompt = f"""You are a professional fact-checking agent. Your evidence has a MultiReflect Reliability Score of {score}/6.0.
            Question: {query}
            Your Document: {document}
            Task: Provide your initial verdict (True, OOC, or MC) based on this document.
            {core_instruction}

            Format: 'Answer: {{}}. Explanation: {{}}.'"""

    return call_llm(prompt, generator)

def aggregate_responses(query: str, responses: List[str], scores: List[float], generator):
    joined = "\n".join([f"Agent {i+1} (Reliability Score: {scores[i]}): {r}" for i, r in enumerate(responses)])
    prompt = f"""You are a master aggregator resolving conflicts between fact-checking agents. 

                Question: {query}
                Agent Debates with Reliability Scores:
                {joined}

                Instructions:
                1. Prioritize agents with higher Reliability Scores.
                2. Resolve any contradictions and choose the most supported label (True, OOC, or MC).
                3. YOU MUST CONCLUDE YOUR RESPONSE WITH THE FOLLOWING FORMAT:
                Final Verdict: <Label>
                Reason: <Brief Explanation>

                Replace <Label> with exactly one of: [True, OOC, MC]."""
    
    return call_llm(prompt, generator)


def multi_agent_debate(query: str, documents: List[str], scores: List[float], generator, num_rounds: int = 3):
    records = {}
    num_agents = len(documents)
    agent_outputs = []

    # Round 1
    records["round1"] = {"answers": [], "explanations": []}
    for i in range(num_agents):
        response = agent_response(query, documents[i], scores[i], generator)
        # 解析 Answer 與 Explanation
        ans_start = response.find("Answer: ") + len("Answer: ")
        exp_start = response.find("Explanation: ") + len("Explanation: ")
        answer = response[ans_start:response.find("Explanation")].strip()
        explanation = response[exp_start:].strip()
        
        records["round1"]["answers"].append(answer)
        records["round1"]["explanations"].append(explanation)
        agent_outputs.append(response)
    
    records["round1"]["aggregation"] = aggregate_responses(query, agent_outputs, scores, generator)
    
    # Additional rounds
    final_aggregation = records["round1"]["aggregation"]
    for t in range(1, num_rounds):
        round_key = f"round{t+1}"
        records[round_key] = {"answers": [], "explanations": []}
        new_outputs = []
        
        for i in range(num_agents):
            history = "\n".join([f"Agent {j+1}: {agent_outputs[j]}" for j in range(num_agents) if j != i])
            
            response = agent_response(query, documents[i], scores[i], generator, history)
            
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
            records[round_key]["aggregation"] = aggregate_responses(query, agent_outputs, scores, generator)
            final_aggregation = records[round_key]["aggregation"]

    records["final_aggregation"] = final_aggregation
    return records


def main():
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

    ranking_dir = os.path.join(args.data_path, "ranking_score")
    if not os.path.exists(ranking_dir):
        print(f"Error: Directory {ranking_dir} does not exist.")
        return

    # 找出所有是數字的資料夾名稱，並按數字大小排序
    folder_indices = sorted([f for f in os.listdir(ranking_dir) if f.isdigit() and int(f) <= 99], key=int)
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

            # score.csv
            df = pd.read_csv(csv_path)
            top_k = df.sort_values(by='total', ascending=False).head(7)
            
            query_with_caption = (
                f"Original Caption: '{original_caption}'\n\n"
                "Task: Verify if this specific caption is [True], [Out-of-Context (OOC)], "
                "or [Miscaptioned (MC)] based on the visual evidence and provided documents."
            )

            result = multi_agent_debate(
                query=query_with_caption, 
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

    print(f"\nAll tasks finished. Total cases processed: {len(all_results)}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()