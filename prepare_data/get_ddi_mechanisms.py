import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

from prepare_data.NLPProcess import NLPProcess
import json
import csv
import re
import Levenshtein
from collections import defaultdict, OrderedDict, Counter


def get_drugbank_mechanisms_list():
    # get all drugbank mechanism templates
    data_files = ["transductive_train.json", "transductive_dev.json", "transductive_test.json"]
    all_exps = []

    for filename in data_files:
        with open(os.path.join("datasets/raw", filename)) as f:
            for line in f.readlines():
                line = json.loads(line)
                if line['label']:
                    all_exps.append(line['drugbank_explanation'])

    mechanisms ,actions ,_ ,_ = NLPProcess(["DRUG1", "DRUG2"], all_exps)

    all_mechs = defaultdict(int)

    for m, a in zip(mechanisms, actions):
        all_mechs[m.lower()+"_"+a.lower()] += 1

    all_mechs = list(all_mechs.items())
    all_mechs.sort(key=lambda x:x[1], reverse=True)

    with open("datasets/drugbank_ddi_mechanisms.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
            quotechar='\"', quoting=csv.QUOTE_MINIMAL) 

        for lid, (mech, cnt) in enumerate(all_mechs):
            m, a  = mech.split("_")
            writer.writerow([lid+1, m, a, cnt])

def process_explanation(text):
    pattern = r'\b(\w*DRUG1\w*)\b'
    text = re.sub(pattern, "DRUG", text, flags=re.IGNORECASE)
    pattern = r'\b(\w*DRUG2\w*)\b'
    text = re.sub(pattern, "DRUG", text, flags=re.IGNORECASE)
    text = re.sub(r'\([^)]*\)', '', text)
    text = ' '.join(text.split())
    
    return text

def get_ddinter_mechanisms_list():
    data_folder = "datasets/raw"
    data_files = ["transductive_train.json", "transductive_dev.json", "transductive_test.json"]
    exp_dict = defaultdict(int)

    for filename in data_files:
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                if not data['label']: continue
                drug1_name = data['drug1_name']
                drug2_name = data['drug2_name']
                explanation = data['ddinter_explanation']
                template = process_explanation(explanation)
                exp_dict[template] += 1


    exp_list = list(exp_dict.items())
    for k in exp_dict.copy():
        if exp_dict[k] == 1:
            exp_dict.pop(k)


    templates = list(exp_dict.keys())

    for i, exp_tuple in enumerate(exp_list):
        exp = exp_tuple[0]
        if exp_tuple[1] == 1:
            scores = [Levenshtein.ratio(exp, sentence) for sentence in templates]
            sim = templates[scores.index(max(scores))]
            levenshetein_score = max(scores)
            if levenshetein_score > 0.9:
                exp_dict[sim] += 1
            else:
                # print(levenshetein_score)
                # print(exp)
                # print(sim)
                exp_dict[exp] += 1

    exp_list = list(exp_dict.items())
    exp_list.sort(key=lambda x:x[1], reverse=True)


    with open("datasets/ddinter_ddi_mechanisms.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        for i, exp_tuple in enumerate(exp_list):
            writer.writerow([i+1, exp_tuple[1], exp_tuple[0]])
    

def read_ddinter_templates():
    templates = {}
    temp_path = "datasets/ddinter_ddi_mechanisms.csv"
    with open(temp_path, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            templates[line[2]] = line[0]
    
    return templates

def update_dataset_with_ddinter_mechanism_label():
    src_folder = "datasets/raw"
    out_folder = "datasets/"
    data_files = os.listdir(src_folder)

    templates = read_ddinter_templates()
    temp_list = list(templates.keys())

    for filename in data_files:
        src_data = []
        with open(os.path.join(src_folder, filename)) as f:
            for line in f.readlines():
                data = json.loads(line)
                explanation = process_explanation(data['ddinter_explanation'])
                if not data['label']:
                    data['ddinter_exp_id'] = "0" 
                elif explanation in templates:
                    data['ddinter_exp_id'] = templates[explanation]
                else:
                    scores = [Levenshtein.ratio(explanation, sentence) for sentence in temp_list]
                    sim = temp_list[scores.index(max(scores))]
                    levenshetein_score = Levenshtein.ratio(explanation, sim)
                    if levenshetein_score > 0.9:
                        data['ddinter_exp_id'] = templates[sim]
                    else:
                        raise Exception("No matching template")

                src_data.append(data)

        with open(os.path.join(out_folder, filename), "w") as f:
            for data in src_data:
                f.write(json.dumps(data)+"\n")


def read_drugbank_templates():
    templates = {}
    temp_path = "datasets/drugbank_ddi_mechanisms.csv"
    with open(temp_path, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            mechanism = "_".join([line[1], line[2]])
            templates[mechanism] = line[0]
    
    return templates

def find_nearest_drugbank_template(query, templates):
    temp_list = [tuple(x.split('_')) for x in templates.keys()]
    query_m, query_a = query.split('_')
    scores = [Levenshtein.ratio(query_m, temp[0]) if query_a == temp[1] else 0 for temp in temp_list]
    sim = temp_list[scores.index(max(scores))]
    return templates["_".join(sim)]


def update_dataset_with_drugbank_mechanism_label():
    src_folder = "datasets/temp"
    out_folder = "datasets/"
    data_files = os.listdir(src_folder)

    templates = read_drugbank_templates()

    for filename in data_files:
    # for filename in ["inductive_test_s1.json"]:
        if filename in ["transductive_train.json", "inductive_test_s1.json"]: continue
        src_data = []
        exps = OrderedDict()
        with open(os.path.join(src_folder, filename)) as f:
            for lid, line in enumerate(f.readlines()):
                data = json.loads(line)
                explanation = data['drugbank_explanation']
                if data['label']:
                    exps[lid] = explanation
                src_data.append(data)

        mechanisms ,actions ,_ ,_ = NLPProcess(["DRUG1", "DRUG2"], list(exps.values()))
        for eid, lid in enumerate(exps.keys()):
            m = mechanisms[eid].lower()
            a = actions[eid].lower()
            exps[lid] = "_".join([m, a])

        for lid, data in enumerate(src_data):
            if not data['label']:
                data['drugbank_exp_id'] = "0"
            else:
                mech = exps[lid]
                if mech in templates:
                    exp_id = templates[mech]
                else:
                    print("%d unfound mech:"%lid, mech)
                    exp_id = find_nearest_drugbank_template(mech, templates)
                data['drugbank_exp_id'] = exp_id

        with open(os.path.join(out_folder, filename), "w") as f:
            for data in src_data:
                f.write(json.dumps(data)+"\n")

def get_ddinter_drugbank_template_mapping():
    # data_files = ["transductive_train.json", "transductive_dev.json", "transductive_test.json"]
    data_files = ["transductive_train.json"]
    ddinter2drugbank = defaultdict(list)

    for filename in data_files:
        with open(os.path.join("datasets", filename), "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                ddinter_exp_id = int(data["ddinter_exp_id"])
                drugbank_exp_id = int(data["drugbank_exp_id"])
                ddinter2drugbank[ddinter_exp_id].append(drugbank_exp_id)

    ddinter2drugbank = list(ddinter2drugbank.items())
    ddinter2drugbank.sort(key=lambda x:x[0])

    with open("datasets/ddinter_to_drugbank_mechanism_map_trans.csv", "w") as f:
        writer = csv.writer(f)
        for ddinter_exp_id, map_list in ddinter2drugbank:
            map_counts = Counter(map_list)
            map_id = map_counts.most_common(1)[0][0]
            writer.writerow([ddinter_exp_id, map_id])




if __name__ == '__main__':
    # get_drugbank_mechanisms_list()
    # get_ddinter_mechanisms_list()
    # update_dataset_with_ddinter_mechanism_label()
    # update_dataset_with_drugbank_mechanism_label()
    get_ddinter_drugbank_template_mapping()
