import os
os.chdir('/Data/projects/Ex_DDI')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(os.path.abspath('.'))
from collections import defaultdict
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import json
import torch
import random
# import nltk
# from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from eval_explanation import compute_metric
import csv
import Levenshtein



def read_templates():
    templates = {}
    temp_path = "data/dataset/explanation_templates.csv"
    with open(temp_path, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            templates[line[2]] = line[0]
    
    return templates

def compute_metric_vote():
    # ref_idx_file = "data/dataset/ref_retrieve_idx/inductive_dev.json"
    # ddi_rtv_idx_file = "data/dataset/retreived_idx/ddi_smiles_dense/top50/inductive_dev.json"
    rtv_idx_file = "data/dataset/retreived_idx/ddi_smiles_molt5_mean/top50/inductive_test_s2.json"
    dev_file = "data/dataset/inductive_test_s2_with_label_revise.json"
    # dev_file = "data/dataset/inductive_test_s1_with_label.json"
    train_file = "data/dataset/inductive_train.json"
    top_k = 9

    rtv_idxs = []
    with open(rtv_idx_file) as f:
        for line in f.readlines():
            rtv_idxs.append(json.loads(line))

    train_data = []
    
    with open(train_file) as f:
        for line in f.readlines():
            train_data.append(json.loads(line))

    templates = read_templates()
    temp_list = list(templates.keys())
    
    y_true = []
    y_pred = []
    exp_true = []
    exp_pos_true = []
    exp_neg_true = []
    exp_pred = []
    exp_pos_pred = []
    exp_neg_pred = []
    exp_label_gold = []
    exp_label_pred = []

    with open(dev_file) as f:
        for lid, line in enumerate(f.readlines()):
            data = json.loads(line)
            label = data['label']
            if os.path.basename(dev_file) == 'inductive_test_s2_with_label_revise.json':
                to_remove = data['to_remove']
                if to_remove:
                    continue

            y_true.append(1 if label else 0)
            exp_true.append(data['explanation'])
            if label:
                exp_pos_true.append(data['explanation'])
            else:
                exp_neg_true.append(data['explanation'])
            # if not label: continue

            rtvs = rtv_idxs[lid][:top_k]
            rtv_labels = []
            for rid, tid in enumerate(rtvs):
                sample = train_data[tid]
                slabel = sample['label']
                rtv_labels.append(1 if slabel else 0)
                if rid == 0:
                    exp_pred.append(sample['explanation'])
                    if label:
                        exp_pos_pred.append(sample['explanation'])
                    else:
                        exp_neg_pred.append(sample['explanation'])
            if 2*sum(rtv_labels) > top_k:
                y_pred.append(1)
                pred = 1
            else:
                y_pred.append(0)
                pred = 0


            # exp matching eval
            if top_k == 1:
                exp_label_gold.append(data['exp_id'])
                explanation = train_data[rtvs[0]]['explanation']
                if not pred:
                    exp_label_pred.append(0) 
                elif explanation in templates:
                    exp_label_pred.append(templates[explanation])
                else:
                    scores = [Levenshtein.ratio(explanation, sentence) for sentence in temp_list]
                    sim = temp_list[scores.index(max(scores))]
                    exp_label_pred.append(templates[sim])


    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", round(accuracy,4))

    # 计算F1值
    f1 = f1_score(y_true, y_pred)
    print("F1 Score:", round(f1, 4))

    # 计算精确率
    precision = precision_score(y_true, y_pred)
    print("Precision:", round(precision, 4))

    # 计算召回率
    recall = recall_score(y_true, y_pred)
    print("Recall:", round(recall, 4))   

    # 计算explanation matching F1
    if top_k == 1:
        exp_micro_f1 = f1_score(exp_label_gold, exp_label_pred, average='micro')
        print("exp_micro_f1:", round(exp_micro_f1, 4))
        exp_macro_f1 = f1_score(exp_label_gold, exp_label_pred, average='macro')
        print("exp_macro_f1:", round(exp_macro_f1, 4))

    # 计算bleu
    if top_k == 1:
        exp_scores = compute_metric(refs=exp_true, preds=exp_pred )
        print("######## Overall Score #########")
        for k, v in exp_scores.items():
            print("%s: %.4f"%(k, v))

        pos_exp_scores = compute_metric(refs=exp_pos_true, preds=exp_pos_pred)
        # pos scores 
        print("######## Positive Case Score #########")
        for k, v in pos_exp_scores.items():
            print("%s: %.4f"%(k, v))

        neg_exp_scores = compute_metric(refs=exp_neg_true, preds=exp_neg_pred)
        # neg scores 
        print("######## Negative Case Score #########")
        for k, v in neg_exp_scores.items():
            print("%s: %.4f"%(k, v))   

def get_reference_indexes():
    train_file = 'data/dataset/inductive_train.json'
    target_file = 'data/dataset/inductive_dev.json'

    # get reference idx
    output_folder = "data/dataset/ref_retrieve_idx/"
    output_file = os.path.join(output_folder, os.path.basename(target_file))


    with open(train_file) as f:
        train_descriptions = []
        for line in f.readlines():
            data = json.loads(line)
            explanation = data['explanation']
            explanation = explanation.replace('DRUG1','DRUG').replace('DRUG2','DRUG').split(" ")
            train_descriptions.append(explanation)
        train_bm25 = BM25Okapi(train_descriptions)

    with open(output_file, 'w') as fout:
        with open(target_file) as f:
            for line in tqdm(f.readlines()):
                retrv_idxs = []
                data = json.loads(line)
                explanation = data['explanation']
                explanation = explanation.replace('DRUG1','DRUG').replace('DRUG2','DRUG').split(" ")
                scores = train_bm25.get_scores(explanation)
                top_rsts = torch.topk(torch.tensor(scores), 200)
                for s, i in zip(top_rsts[0].tolist(), top_rsts[1].tolist()):
                    retrv_idxs.append((s, i))
                fout.write(json.dumps(retrv_idxs)+"\n")    

def sample_retrieved_case():
    ref_idx_file = "data/dataset/ref_retrieve_idx/inductive_dev.json"
    rtv_idx_file = "data/dataset/retreived_idx/ddi_smiles_dense_molt5/top50/inductive_dev.json"
    dev_file = "data/dataset/inductive_dev.json"
    train_file = "data/dataset/inductive_train.json"

    ref_idx = []
    with open(ref_idx_file) as f:
        for line in f.readlines():
            ref_idx.append(json.loads(line))

    rtv_idx = []
    with open(rtv_idx_file) as f:
        for line in f.readlines():
            rtv_idx.append(json.loads(line))

    train_data = []
    with open(train_file) as f:
        for line in f.readlines():
            train_data.append(json.loads(line))

    with open("./retrieval/ddi_smiles_dense_molt5_sample.txt", "w") as fout:
        with open(dev_file) as f:
            lines = f.readlines()
            # sample_ids = random.sample(range(len(lines)), 10)
            sample_ids = [20863, 7302, 5264, 19771, 6480, 22065, 10093, 17579, 15932, 8765]
            print(sample_ids)

            for lid, line in enumerate(lines):
                if lid not in sample_ids: continue
                data = json.loads(line)
                rtvs = rtv_idx[lid]
                rtv_sample = []
                for tid in rtvs[:10]:
                    rtv_sample.append(train_data[tid])

                ref_sample = []
                for ref in ref_idx[lid][:10]:
                    ref_sample.append(train_data[ref[1]])

                fout.write("\n")
                fout.write("%d QUERY: drug1_name: %s, drug2_name: %s, %d\n"%(lid, data['drug1_name'], data['drug2_name'], int(data['label'])))
                fout.write("explanation: %s\n"%data['explanation'])

                # fout.write("\nReference:\n")
                # for sid, sample in enumerate(ref_sample):
                #     fout.write("%d: drug1_name: %s, drug2_name: %s, %d\n"%(sid, sample['drug1_name'], sample['drug2_name'], int(sample['label'])))
                #     fout.write("explanation: %s\n"%sample['explanation'])
                #     fout.write("\n")

                fout.write("\nDDI_SMILES_DENSE_MOLT5 Retrieved:\n")
                for sid, sample in enumerate(rtv_sample):
                    fout.write("%d: drug1_name: %s, drug2_name: %s, %d\n"%(sid, sample['drug1_name'], sample['drug2_name'], int(sample['label'])))
                    fout.write("explanation: %s\n"%sample['explanation'])
                    fout.write("\n")

                

if __name__ == '__main__':

    compute_metric_vote()
    # sample_retrieved_case()
