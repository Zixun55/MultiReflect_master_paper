import json
import pandas as pd
from urllib.parse import urlparse
from ast import literal_eval

df = pd.read_csv('./ranking/annotated_sources_final.csv')

def get_factuality(hostname):
    df_selected = df[df['url'].str.contains(hostname) | df['merged_urls'].str.contains(hostname)]
    if len(df_selected) > 0:
        point_map = {'Very High Factuality' : 1.0, 'High Factuality' : 0.66, 'Mostly Factual': 0.33, 'Mixed Factuality' : 0.0, 'Low Factuality' : -0.33, 'Very Low Factuality' : -0.66, 'Satire': -1.0}
        if df_selected['factuality (MBFC)'].iloc[0] in point_map:
            return point_map[df_selected['factuality (MBFC)'].iloc[0]]
        else:
            return 0.0
    else:
        return 0.0
    
def get_reliability(hostname):
    df_selected = df[df['url'].str.contains(hostname) | df['merged_urls'].str.contains(hostname)]
    if len(df_selected) > 0:
        point_map = {'Reliable, Analysis/Fact Reporting' : 1.0, 'Generally Reliable/Analysis OR Other Issues': 0.5, 'Mixed Reliability/Opinion OR Other Issues': 0.0}
        if df_selected['reliability (AF)'].iloc[0] in point_map:
            return point_map[df_selected['reliability (AF)'].iloc[0]]
        else:
            return 0.0
    else:
        return 0.0

def get_bias(hostname):
    df_selected = df[df['url'].str.contains(hostname) | df['merged_urls'].str.contains(hostname)]
    if len(df_selected) > 0:
        point_map = {'Left': 0.0, 'Left-Center': 0.5, 'Center': 1.0, 'Right-Center': 0.5, 'Right': 0.0}
        if df_selected['bias (final)'].iloc[0] in point_map:
            return point_map[df_selected['bias (final)'].iloc[0]]
        else:
            return 0.0
    else:
        return 0.0
    
def get_authoritative_scores(file_name):
    evidence_file = json.load(open(f'./data/filtered/{file_name}/text_data/paragraphs.json'))
    data = {}
    for key in evidence_file:
        data_src = []
        for evidence in evidence_file[key]:
            if evidence['link'] is not None:
                parsed_url = urlparse(evidence['link'])
                final_url = parsed_url.netloc.replace('www.', '')
                fact = get_factuality(final_url)
                reliability = get_reliability(final_url)
                bias = get_bias(final_url)
                data_src.append({'factuality': fact, 'reliability': reliability, 'bias': bias})
            else:
                data_src.append({'factuality': 0.0, 'reliability': 0.0, 'bias': 0.0})
        data[key] = data_src
    return data