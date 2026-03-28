import json
import os
import dateutil.parser
from datetime import datetime
from utils import get_llava_cot_response

def get_claim_type(caption, client):
    instruction = (
        "Task: Classify the following Input Text into one of two categories for fact-checking time evaluation.\n\n"
        "Categories:\n"
        "- [Knowledge]: Information that is updated over time (e.g., current population, latest GDP, current president, rankings).\n"
        "- [Event]: A specific occurrence at a particular time point (e.g., a specific protest, a car accident, a quote from yesterday).\n\n"
        f"Input Text: {caption}\n\n"
        "Output: Start your response with either [Knowledge] or [Event]."
    )
    
    prompt = {
        "messages": [{"role": "user", "content": instruction}],
        "images": []
    }
    
    try:
        response = get_llava_cot_response(prompt, client)
        if '[Knowledge]' in response:
            return 'knowledge'
        return 'event'
    except:
        return 'event'

def get_freshness_scores(file_name, caption, eval_type="proposed", client=None):
    """
    eval_type: 
      - "original": 舊架構，寫死 2022-01-01
      - "proposed": 新架構
    """
    path = f'./data/filtered/{file_name}/text_data/paragraphs.json'
    if not os.path.exists(path):
        return {}
        
    evidence_file = json.load(open(path))
    data = {}
    
    if eval_type == "proposed" and client:
        claim_type = "event"
        claim_type = get_claim_type(caption, client)
        print("Claim Type:", claim_type)

    # 基準日期設定(原架構的設定)
    baseline_date = dateutil.parser.parse('2022-01-01').replace(tzinfo=None)
    # 假設目前的參考時間
    current_date = datetime.now().replace(tzinfo=None) 

    for key in evidence_file:
        print(f"Processing evidence step: 1")
        data_src = []
        for evidence in evidence_file[key]:
            print(f"Processing evidence step: 2")
            try:
                if evidence.get('timestamp') is not None:
                    ev_date = dateutil.parser.parse(evidence['timestamp']).replace(tzinfo=None)
                    
                    if eval_type == "original":
                        score = 1.0 if ev_date > baseline_date else 0.0
                    
                    else:
                        print(f"Evaluating evidence with timestamp: {ev_date}")
                        if claim_type == "knowledge":
                            # 知識類：越新越高分
                            diff_years = abs((ev_date - current_date).days) / 365.0
                            if diff_years < 2:
                                score = 1 - (diff_years / 2)
                            else:
                                score = 0.0
                        else:
                            # 事件類：證據時間與當前/claim時間的差距天數
                            diff_years = abs((ev_date - baseline_date).days) / 365.0
                            if diff_years < 2 and ev_date > baseline_date:
                                score = 1 - (diff_years / 2)
                            else:
                                score = 0.0
                        score = score * 2
                            
                    data_src.append(round(score, 2))
                else:
                    print("No timestamp found")
                    data_src.append(0.0)
            except:
                data_src.append(0.0)
                continue
        data[key] = data_src
        
    return data