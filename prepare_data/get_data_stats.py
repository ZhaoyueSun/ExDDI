from collections import defaultdict
import os
import json

def get_stat():
    data_folder = "datasets/cross_validation/fold5"
    data_files = ["transductive_train.json", "transductive_dev.json", "transductive_test.json", 
    "inductive_train.json", "inductive_dev.json", "inductive_test_s1.json", "inductive_test_s2.json"]
    data_file = data_files[6]

    drug_cnt = defaultdict(int)
    drugbank_exp_cnt = defaultdict(int)
    ddinter_exp_cnt = defaultdict(int)
    pos_cnt = 0
    neg_cnt = 0

    with open(os.path.join(data_folder, data_file), 'r') as f:
        for line in f.readlines():
            data = json.loads(line)

            drug_cnt[data['drug1_id']] += 1
            drug_cnt[data['drug2_id']] += 1
            if data['label']:
                drugbank_exp_cnt[data['drugbank_exp_id']] += 1
                ddinter_exp_cnt[data['ddinter_exp_id']] += 1
                pos_cnt += 1
            else:
                neg_cnt += 1
            
    # print("Pos Cnt:", pos_cnt)
    # print("Neg Cnt:", neg_cnt)
    print("DDIs:", pos_cnt + neg_cnt)
    print("Drugs:", len(drug_cnt))
    print("Drugbank Exps:", len(drugbank_exp_cnt))
    print("DDInter Exps:", len(ddinter_exp_cnt))

if __name__ == '__main__':
    get_stat()