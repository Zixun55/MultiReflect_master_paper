import json
import dateutil

def get_freshness_scores(file_name):
    evidence_file = json.load(open(f'./data/filtered/{file_name}/text_data/paragraphs.json'))
    data = {}
    for key in evidence_file:
        data_src = []
        for evidence in evidence_file[key]:
            try:
                if evidence['timestamp'] is not None:
                    date = dateutil.parser.parse(evidence['timestamp'])
                    date = date.replace(tzinfo=None)
                    if date > dateutil.parser.parse('2022-01-01'):
                        data_src.append(1.0)
                    else:
                        data_src.append(0.0)
                else:
                    data_src.append(0.0)
            except:
                data_src.append(0.0)
                continue
        data[key] = data_src
    return data