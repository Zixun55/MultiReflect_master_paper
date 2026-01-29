import ranking.relevance as relevance
import ranking.support as support
import ranking.useful as useful
import ranking.authoritative as authoritative
import ranking.freshness as freshness
import os
import json
import pandas as pd
import csv

def get_relevance_score(text):
    if '[Relevant]' in text:
        return 1
    elif '[Irrelevant]' in text:
        return 0
    else:
        if 'Relevant' in text:
            return 1
        elif 'Irrelevant' in text:
            return 0
        else:
            print('Error: relevance score not found', text)
            return 0

def get_support_score(text):
    if '[Fully supported]' in text:
        return 1
    elif '[Partially supported]' in text:
        return 0.5
    elif '[No support / Contradictory]' in text:
        return 0
    else:
        if 'Fully supported' in text:
            return 1
        elif 'Partially supported' in text:
            return 0.5
        elif 'No support / Contradictory' in text:
            return 0
        else:
            print('Error: support score not found', text)
            return 0

def get_usefulness_score(text):
    final_text = text.split('\n')[0]
    if '5' in final_text:
        return 1.0
    elif '4' in final_text:
        return 0.5
    elif '3' in final_text:
        return 0.0
    elif '2' in final_text:
        return -0.5
    elif '1' in final_text:
        return -1.0
    else:
        for i in text.split('\n'):
            if '5' in i:
                return 1.0
            elif '4' in i:
                return 0.5
            elif '3' in i:
                return 0.0
            elif '2' in i:
                return -0.5
            elif '1' in i:
                return -1.0
            else:
                continue
    return 0

def get_text_scores(file_name, caption, image_path, client):
    text_rel = relevance.get_text_to_image_relevance_sample(file_name, image_path, client)
    text_sup = support.get_text_support_sample(file_name, caption, image_path, client)
    text_use = useful.get_text_useful_sample(file_name, caption, image_path, client)
    text_aut = authoritative.get_authoritative_scores(file_name)
    text_fresh = freshness.get_freshness_scores(file_name)
    
    if not os.path.exists(f'./data/ranking_score/{file_name}/text_data'):
        os.makedirs(f'./data/ranking_score/{file_name}/text_data')
    with open(f'./data/ranking_score/{file_name}/text_data/relevance.json', 'w') as f:
        json.dump(text_rel, f)
    with open(f'./data/ranking_score/{file_name}/text_data/support.json', 'w') as f:
        json.dump(text_sup, f)
    with open(f'./data/ranking_score/{file_name}/text_data/useful.json', 'w') as f:
        json.dump(text_use, f)
    with open(f'./data/ranking_score/{file_name}/text_data/authoritative.json', 'w') as f:
        json.dump(text_aut, f)
    with open(f'./data/ranking_score/{file_name}/text_data/freshness.json', 'w') as f:
        json.dump(text_fresh, f)
    
    evidence = json.load(open(f'./data/filtered/{file_name}/text_data/paragraphs.json'))
    final_metrics = []
    for k, v in evidence.items():
        for idx in range(len(v)):
            metrics = {}

            try:
                raw_text = v[idx].get('text', '')
                clean_text = " ".join(raw_text.split()).replace('"', "'")
                metrics['evidence'] = clean_text
            except:
                metrics['evidence'] = ''

            try:
                metrics['relevance'] = get_relevance_score(text_rel[k][idx])
            except:
                metrics['relevance'] = 0
            try:
                metrics['support'] = get_support_score(text_sup[k][idx])
            except:
                metrics['support'] = 0
            try:
                metrics['usefulness'] = get_usefulness_score(text_use[k][idx])
            except:
                metrics['usefulness'] = 0
            if metrics['relevance'] > 0 or metrics['support'] > 0 or metrics['usefulness'] > 0:
                try:
                    metrics['factuality'] = text_aut[k][idx]['factuality']
                except:
                    metrics['factuality'] = 0
                try:
                    metrics['reliability'] = text_aut[k][idx]['reliability']
                except:
                    metrics['reliability'] = 0
                try:
                    metrics['bias'] = text_aut[k][idx]['bias']
                except:
                    metrics['bias'] = 0
                try:
                    metrics['freshness'] = text_fresh[k][idx]
                except:
                    metrics['freshness'] = 0
            else:
                metrics['factuality'] = 0
                metrics['reliability'] = 0
                metrics['bias'] = 0
                metrics['freshness'] = 0
            metrics['total'] = metrics['relevance'] + metrics['support'] + metrics['usefulness'] + metrics['factuality'] + metrics['reliability'] + metrics['bias'] + metrics['freshness']
            final_metrics.append(metrics)
            
    df = pd.DataFrame(final_metrics)
    df.to_csv(f'./data/ranking_score/{file_name}/text_data/final_scores.csv', index=False)
    
def get_image_scores(file_name, caption, image_path, client):
    image_rel = relevance.get_image_to_text_relevance_sample(file_name, caption, client)
    image_sup = support.get_image_support_sample(file_name, caption, image_path, client)
    image_use = useful.get_image_useful_sample(file_name, caption, image_path, client)
    
    if not os.path.exists(f'./data/ranking_score/{file_name}/image_data'):
        os.makedirs(f'./data/ranking_score/{file_name}/image_data')
    with open(f'./data/ranking_score/{file_name}/image_data/relevance.json', 'w') as f:
        json.dump(image_rel, f)
    with open(f'./data/ranking_score/{file_name}/image_data/support.json', 'w') as f:
        json.dump(image_sup, f)
    with open(f'./data/ranking_score/{file_name}/image_data/useful.json', 'w') as f:
        json.dump(image_use, f)
    
    evidence = os.listdir(f'./data/filtered/{file_name}/image_data/')
    final_metrics = []
    for idx in range(len(evidence)):
        metrics = {}
        try:
            metrics['evidence'] = evidence[idx]
        except:
            metrics['evidence'] = ''
        try:
            metrics['relevance'] = get_relevance_score(image_rel[evidence[idx]])
        except:
            metrics['relevance'] = 0
        try:
            metrics['support'] = get_support_score(image_sup[evidence[idx]])
        except:
            metrics['support'] = 0
        try:
            metrics['usefulness'] = get_usefulness_score(image_use[evidence[idx]])
        except:
            metrics['usefulness'] = 0
        metrics['total'] = metrics['relevance'] + metrics['support'] + metrics['usefulness']
        final_metrics.append(metrics)
    
    df = pd.DataFrame(final_metrics)
    df.to_csv(f'./data/ranking_score/{file_name}/image_data/final_scores.csv', index=False)