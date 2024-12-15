import re
import json
import csv
import random
import os

def get_drug_smiles_map():

    drug2smiles = {}

    with open("data/DDInter/drug_description.jsonl", 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            drug_id = data["drug_id"]
            smiles = data["smiles"]
            if smiles != "None":
                drug2smiles[drug_id] = smiles

    print("Total drugs with SMILES:", len(drug2smiles))

    return drug2smiles

def get_drug_id_name_map():

    drug2name = {}

    with open("data/DDInter/DDInter_vocab.csv") as f:
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        for row in reader:
            drug2name[row[0]] = row[1]

    return drug2name

def get_drug_description_map():
    
    drug2desc = {}

    with open("data/DDInter/drug_description.jsonl") as f:
        for line in f.readlines():
            data = json.loads(line)
            drug_id = data["drug_id"]
            description = data["description"]
            drug2desc[drug_id] = description

    return drug2desc


def get_drugbank_ddi_descriptions():
    # read mappings
    drugbank2ddinter = {}
    with open("data/DDInter/DDInter_Drugbank_mapping.csv") as f:
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        for row in reader:
            drugbank2ddinter[row[2]] = row[0]

    # read drugbank ddi list:
    ddi_list = {}
    with open("data/drugBank/drugbank_ddis.csv") as f:
        reader = csv.reader(f, delimiter='\t', quotechar='\"')
        for row in reader:
            if row[1] not in drugbank2ddinter or row[3] not in drugbank2ddinter:
                continue
            pos_key = "_".join([drugbank2ddinter[row[1]], drugbank2ddinter[row[3]]])
            ddi_list[pos_key] = row[5]

    return ddi_list



def preprocess_ddi_explanation(explanation, drug1_name, drug2_name):
    # replace mentions of drug_name to smiles representation.
    pattern = r'\b(\w*{}\w*)\b'.format(drug1_name)
    explanation = re.sub(pattern, "DRUG1", explanation, flags=re.I)
    pattern = r'\b(\w*{}\w*)\b'.format(drug2_name)
    explanation = re.sub(drug2_name, "DRUG2", explanation, flags=re.I)

    if '(' in drug1_name:
        drug1_name = drug1_name.split('(')[0].strip()
        pattern = r'\b(\w*{}\w*)\b'.format(drug1_name)
        explanation = re.sub(pattern, "DRUG1", explanation, flags=re.I)

    if '(' in drug2_name:
        drug2_name = drug2_name.split('(')[0].strip()
        pattern = r'\b(\w*{}\w*)\b'.format(drug2_name)
        explanation = re.sub(pattern, "DRUG2", explanation, flags=re.I)

    # explanation = explanation.replace('\\\\', '\\')

    return explanation

def collect_positives():

    drug2smiles = get_drug_smiles_map()
    drug2name = get_drug_id_name_map()
    drugbank_ddis = get_drugbank_ddi_descriptions()
    output = []
    filter_by_name = 0
    filter_by_smiles = 0
    filter_by_no_exp = 0

    filtered_ddis = {}
    with open("data/DDInter/DDInter_ddi_list_filtered.csv", "r") as f:
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        for row in reader:
            k = "_".join([row[0], row[1]])
            filtered_ddis[k] = True
    

    with open("data/DDInter/ddi_description.jsonl", "r") as f:
        for line in f.readlines():
            record = json.loads(line)
            drug1_id = record["drug1"]
            drug2_id = record["drug2"]

            if "_".join([drug1_id, drug2_id]) not in filtered_ddis and "_".join([drug2_id, drug1_id]):
                continue

            if drug1_id not in drug2smiles or drug2_id not in drug2smiles:
                filter_by_smiles += 1
                continue

            if drug1_id not in drug2name or drug2_id not in drug2name:
                filter_by_name += 1
                continue

            severity = record["tags"][0]
            mechanism = record["tags"][1:]
            drug1_name = drug2name[drug1_id]
            drug2_name = drug2name[drug2_id]
            drug1_smiles = drug2smiles[drug1_id]
            drug2_smiles = drug2smiles[drug2_id]
            label = True
            ddinter_explanation = record["description"]
            if not ddinter_explanation:
                filter_by_no_exp += 1
                continue

            ddinter_explanation = preprocess_ddi_explanation(ddinter_explanation, drug1_name, drug2_name)

            if "_".join([drug1_id, drug2_id]) in drugbank_ddis:
                drugbank_explanation = drugbank_ddis["_".join([drug1_id, drug2_id])]
            elif "_".join([drug2_id, drug1_id]) in drugbank_ddis:
                drugbank_explanation = drugbank_ddis["_".join([drug2_id, drug1_id])]
            else:
                raise Exception("Not found the ddi in drugbank!")

            drugbank_explanation = preprocess_ddi_explanation(drugbank_explanation, drug1_name, drug2_name)

            new = {
                'drug1_id': drug1_id,
                'drug1_name': drug1_name,
                'drug1_smiles': drug1_smiles,
                'drug2_id': drug2_id,
                'drug2_name': drug2_name,
                'drug2_smiles': drug2_smiles,
                'label': label,
                'severity': severity,
                'mechanism': mechanism,
                'ddinter_explanation': ddinter_explanation,
                'drugbank_explanation': drugbank_explanation
            }

            output.append(new)

    print("total ddi explanations:", len(output))
    print("filter_by_name", filter_by_name)
    print("filter_by_smiles", filter_by_smiles)
    print("filter_by_no_exp", filter_by_no_exp)

    with open("datasets/all_positives.json", "w") as f:
        for rec in output:
            f.write(json.dumps(rec)+"\n")


def collect_negatives():

    drug2smiles = get_drug_smiles_map()
    drug2name = get_drug_id_name_map()
    drug2desc = get_drug_description_map()

    positive_ids = []
    
    with open("datasets/all_positives.json", "r") as f:
        lines = f.readlines()
        positive_len = len(lines)
        for line in lines:
            record = json.loads(line)
            drug1_id = record["drug1_id"]
            drug2_id = record["drug2_id"]
            positive_ids.append(drug1_id)
            positive_ids.append(drug2_id)

    positive_ids = set(positive_ids)

    negative_pairs = []

    with open("data/DDInter/DDInter_neg_list_filtered.csv", "r") as f:
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        for row in reader:
            if row[0] in positive_ids and row[1] in positive_ids:
                negative_pairs.append([row[0], row[1]])

    print("positive pairs:", positive_len)
    print("negative pairs:", len(negative_pairs))

    random.shuffle(negative_pairs)
    # negative_pairs = negative_pairs[:positive_len]

    with open("datasets/all_negatives.json", "w") as f:
        for drug1_id, drug2_id in negative_pairs:
            drug1_name = drug2name[drug1_id]
            drug2_name = drug2name[drug2_id]
            drug1_smiles = drug2smiles[drug1_id]
            drug2_smiles = drug2smiles[drug2_id]
            drug1_desc = drug2desc[drug1_id].strip()
            drug2_desc = drug2desc[drug2_id].strip()

            severity = None
            mechanism = []
            label = False

            explanation = "%s %s There were no known direct interactions reported between them."%(drug1_desc, drug2_desc)
            explanation = preprocess_ddi_explanation(explanation, drug1_name, drug2_name)

            new = {
                'drug1_id': drug1_id,
                'drug1_name': drug1_name,
                'drug1_smiles': drug1_smiles,
                'drug2_id': drug2_id,
                'drug2_name': drug2_name,
                'drug2_smiles': drug2_smiles,
                'label': label,
                'severity': severity,
                'mechanism': mechanism,
                'ddinter_explanation': explanation,
                'drugbank_explanation': explanation
            }

            f.write(json.dumps(new)+"\n")


def split_transductive():
    """
    split the dataset in transductive setting:  drugs in the test set may be included in the training set, but DDIs cannot appear in the training set.
    split by pairs: 7/1/2 train/dev/test
    """
    train_set = []
    dev_set = []
    test_set = []

    with open("datasets/all_positives.json") as f:
        all_positives = f.readlines()
        len_pos = len(all_positives)

    random.shuffle(all_positives)

    train_set += all_positives[:int(len_pos*0.7)]
    dev_set += all_positives[int(len_pos*0.7):int(len_pos*0.8)]
    test_set += all_positives[int(len_pos*0.8):]

    with open("datasets/all_negatives.json") as f:
        all_negatives = f.readlines()
        # assert(len_pos == len(all_negatives))

    random.shuffle(all_negatives)

    train_set += all_negatives[:int(len_pos*0.7)]
    dev_set += all_negatives[int(len_pos*0.7):int(len_pos*0.8)]
    test_set += all_negatives[int(len_pos*0.8):int(len_pos)]

    print("training samples:", len(train_set))
    print("develop samples:", len(dev_set))
    print("test samples:", len(test_set))

    with open("datasets/transductive_train.json", "w") as f:
        random.shuffle(train_set)
        f.writelines(train_set)

    with open("datasets/transductive_dev.json", "w") as f:
        random.shuffle(dev_set)
        f.writelines(dev_set)

    with open("datasets/transductive_test.json", "w") as f:
        random.shuffle(test_set)
        f.writelines(test_set)
    

def split_inductive():
    """
    split the dataset in the inductive setting: 20% drugs reserved for test set (D3), 5% drugs (D2) reserved for dev set, others (D1) for training
    Training set: both drugs appear in D1
    Dev:  both drugs appear in D2 or one drug in D1, one drug in D2
    Test_S1: both drugs in D3,
    Test_S2: one drug in D3, one drug in D1
    """

    drug_ids = []
    all_positives = []
    all_negatives = []
    # get all drug_ids
    with open("datasets/all_positives.json") as f:
        for line in f.readlines():
            data = json.loads(line)
            all_positives.append(data)
            drug_ids.append(data['drug1_id'])
            drug_ids.append(data['drug2_id'])

    with open("datasets/all_negatives.json") as f:
        for line in f.readlines():
            data = json.loads(line)
            all_negatives.append(data)

    drug_ids = list(set(drug_ids))
    random.shuffle(drug_ids)
    test_drugs = drug_ids[:int(len(drug_ids)*0.2)]
    dev_drugs = drug_ids[int(len(drug_ids)*0.2):int(len(drug_ids)*0.25)]
    train_drugs = drug_ids[int(len(drug_ids)*0.25):]
    

    train_set = []
    dev_set = []
    test_s1 = []
    test_s2 = []
    unuse_pos = 0
    unuse_neg = 0

    for data in all_positives:
        if data['drug1_id'] in train_drugs and data['drug2_id'] in train_drugs:
            train_set.append(data)
        elif data['drug1_id'] in dev_drugs and data['drug2_id'] in dev_drugs:
            dev_set.append(data)
        elif (data['drug1_id'] in train_drugs and data['drug2_id'] in dev_drugs) or (data['drug2_id'] in train_drugs and data['drug1_id'] in dev_drugs):
            dev_set.append(data)
        elif data['drug1_id'] in test_drugs and data['drug2_id'] in test_drugs:
            test_s1.append(data)
        elif (data['drug1_id'] in train_drugs and data['drug2_id'] in test_drugs) or (data['drug2_id'] in train_drugs and data['drug1_id'] in test_drugs):
            test_s2.append(data)
        else:
            unuse_pos += 1

        

    print("train positive pairs:", len(train_set))
    print("dev positive pairs:", len(dev_set))
    print("test s1 positive pairs:", len(test_s1))
    print("test s2 positive pairs:", len(test_s2))
    train_pos = len(train_set)
    dev_pos = len(dev_set)
    test1_pos = len(test_s1)
    test2_pos = len(test_s2)
    
    train_neg, dev_neg, test1_neg, test2_neg = 0,0,0,0
        
    for data in all_negatives:
        if data['drug1_id'] in train_drugs and data['drug2_id'] in train_drugs:
            if train_neg < train_pos:
                train_set.append(data)
                train_neg += 1
        elif data['drug1_id'] in dev_drugs and data['drug2_id'] in dev_drugs:
            if dev_neg < dev_pos:
                dev_set.append(data)
                dev_neg += 1
        elif (data['drug1_id'] in train_drugs and data['drug2_id'] in dev_drugs) or (data['drug2_id'] in train_drugs and data['drug1_id'] in dev_drugs):
            if dev_neg < dev_pos:
                dev_set.append(data)
                dev_neg += 1
        elif data['drug1_id'] in test_drugs and data['drug2_id'] in test_drugs:
            if test1_neg < test1_pos:
                test_s1.append(data)
                test1_neg += 1
        elif (data['drug1_id'] in train_drugs and data['drug2_id'] in test_drugs) or (data['drug2_id'] in train_drugs and data['drug1_id'] in test_drugs):
            if test2_neg < test2_pos:
                test_s2.append(data)
                test2_neg += 1
        else:
            unuse_neg += 1

    print("train pairs:", len(train_set))
    print("dev pairs:", len(dev_set))
    print("test s1 pairs:", len(test_s1))
    print("test s2 pairs:", len(test_s2))
    print("unuse_pos:", unuse_pos)
    print("unuse_neg:", unuse_neg)


    with open("datasets/inductive_train.json", "w") as f:
        random.shuffle(train_set)
        for data in train_set:
            f.write(json.dumps(data)+"\n")

    with open("datasets/inductive_dev.json", "w") as f:
        random.shuffle(dev_set)
        for data in dev_set:
            f.write(json.dumps(data)+"\n")

    with open("datasets/inductive_test_s1.json", "w") as f:
        random.shuffle(test_s1)
        for data in test_s1:
            f.write(json.dumps(data)+"\n")

    with open("datasets/inductive_test_s2.json", "w") as f:
        random.shuffle(test_s2)
        for data in test_s2:
            f.write(json.dumps(data)+"\n")



def prepare_cross_valid_transductive():
    # get original train + dev
    train_file_1 = "datasets/cross_validation/fold1/transductive_train.json"
    dev_file_1 = "datasets/cross_validation/fold1/transductive_dev.json"
    test_file_1 = "datasets/cross_validation/fold1/transductive_test.json"

    data_test1 = []
    with open(test_file_1) as f:
        for line in f.readlines():
            data_test1.append(json.loads(line))
    print(len(data_test1))

    data_combine1 = [] # train + dev of fold1
    with open(train_file_1) as f:
        for line in f.readlines():
            data_combine1.append(json.loads(line))
    with open(dev_file_1) as f:
        for line in f.readlines():
            data_combine1.append(json.loads(line))
    print(len(data_combine1))

    # split data_combine1 to 4 splits
    total = len(data_combine1)
    data_test2 = data_combine1[:int(total*0.25)]
    data_combine2 = data_test1 + data_combine1[int(total*0.25):]

    data_test3 = data_combine1[int(total*0.25):int(total*0.5)]
    data_combine3 = data_test1 + data_test2 + data_combine1[int(total*0.5):]

    data_test4 = data_combine1[int(total*0.5):int(total*0.75)]
    data_combine4 = data_test1 + data_test2 + data_test3 + data_combine1[int(total*0.75):]

    data_test5 = data_combine1[int(total*0.75):]
    data_combine5 = data_test1 + data_test2 + data_test3 + data_test4

    for split in [2,3,4,5]:
        test_data = eval("data_test%d"%split)
        combine_data = eval("data_combine%d"%split)
        dev_data = combine_data[:int(len(combine_data)*0.125)]
        train_data = combine_data[int(len(combine_data)*0.125):]

        print("fold", split)
        print("len train", len(train_data))
        print("len dev", len(dev_data))
        print("len test", len(test_data))

        with open("datasets/cross_validation/fold%d/transductive_train.json"%split, "w") as f:
            for line in train_data:
                f.write(json.dumps(line)+"\n")

        with open("datasets/cross_validation/fold%d/transductive_dev.json"%split, "w") as f:
            for line in dev_data:
                f.write(json.dumps(line)+"\n")

        with open("datasets/cross_validation/fold%d/transductive_test.json"%split, "w") as f:
            for line in test_data:
                f.write(json.dumps(line)+"\n")


def prepare_cross_valid_inductive():
    drug_ids = []
    all_positives = []
    all_negatives = []
    # get all drug_ids
    for filename in ['inductive_train.json', 'inductive_test_s1.json', 'inductive_test_s2.json', 'inductive_dev.json']:
        with open(os.path.join("datasets/cross_validation/fold1", filename)) as f:
            for line in f.readlines():
                line = json.loads(line)
                drug_ids.append(line['drug1_id'])
                drug_ids.append(line['drug2_id'])
                if line['label']:
                    all_positives.append(line)
                else:
                    all_negatives.append(line)

    print("all pos", len(all_positives))
    print("all neg", len(all_negatives))

    drug_ids = set(drug_ids)
    drug_test1 = []
    with open("datasets/cross_validation/fold1/inductive_test_s1.json") as f:
        for line in f.readlines():
            line = json.loads(line)
            drug_test1.append(line['drug1_id'])
            drug_test1.append(line['drug2_id'])
    drug_test1 = set(drug_test1)
    drug_combine1 = list(drug_ids - drug_test1)
    random.shuffle(drug_combine1)

    len_combine = len(drug_combine1)
    drug_test2 = drug_combine1[:int(len_combine*0.25)]
    drug_combine2 = list(drug_ids - set(drug_test2))

    drug_test3 = drug_combine1[int(len_combine*0.25):int(len_combine*0.5)]
    drug_combine3 = list(drug_ids - set(drug_test3))

    drug_test4 = drug_combine1[int(len_combine*0.5):int(len_combine*0.75)]
    drug_combine4 = list(drug_ids - set(drug_test4))

    drug_test5 = drug_combine1[int(len_combine*0.75):]
    drug_combine5 = list(drug_ids - set(drug_test5))

    for split in [2,3,4,5]:
        test_drugs = eval("drug_test%d"%split)
        combine_drugs = eval("drug_combine%d"%split)
        dev_drugs = combine_drugs[:int(len(combine_drugs)*0.0625)]
        train_drugs = combine_drugs[int(len(combine_drugs)*0.0625):]

        train_set = []
        dev_set = []
        test_s1 = []
        test_s2 = []
        unuse_pos = 0
        unuse_neg = 0

        for data in all_positives:
            if data['drug1_id'] in train_drugs and data['drug2_id'] in train_drugs:
                train_set.append(data)
            elif data['drug1_id'] in dev_drugs and data['drug2_id'] in dev_drugs:
                dev_set.append(data)
            elif (data['drug1_id'] in train_drugs and data['drug2_id'] in dev_drugs) or (data['drug2_id'] in train_drugs and data['drug1_id'] in dev_drugs):
                dev_set.append(data)
            elif data['drug1_id'] in test_drugs and data['drug2_id'] in test_drugs:
                test_s1.append(data)
            elif (data['drug1_id'] in train_drugs and data['drug2_id'] in test_drugs) or (data['drug2_id'] in train_drugs and data['drug1_id'] in test_drugs):
                test_s2.append(data)
            else:
                unuse_pos += 1

        print("fold %d train positive pairs:"%split, len(train_set))
        print("fold %d dev positive pairs:"%split, len(dev_set))
        print("fold %d test s1 positive pairs:"%split, len(test_s1))
        print("fold %d test s2 positive pairs:"%split, len(test_s2))
        train_pos = len(train_set)
        dev_pos = len(dev_set)
        test1_pos = len(test_s1)
        test2_pos = len(test_s2)
        
        train_neg, dev_neg, test1_neg, test2_neg = 0,0,0,0
            
        for data in all_negatives:
            if data['drug1_id'] in train_drugs and data['drug2_id'] in train_drugs:
                if train_neg < train_pos:
                    train_set.append(data)
                    train_neg += 1
            elif data['drug1_id'] in dev_drugs and data['drug2_id'] in dev_drugs:
                if dev_neg < dev_pos:
                    dev_set.append(data)
                    dev_neg += 1
            elif (data['drug1_id'] in train_drugs and data['drug2_id'] in dev_drugs) or (data['drug2_id'] in train_drugs and data['drug1_id'] in dev_drugs):
                if dev_neg < dev_pos:
                    dev_set.append(data)
                    dev_neg += 1
            elif data['drug1_id'] in test_drugs and data['drug2_id'] in test_drugs:
                if test1_neg < test1_pos:
                    test_s1.append(data)
                    test1_neg += 1
            elif (data['drug1_id'] in train_drugs and data['drug2_id'] in test_drugs) or (data['drug2_id'] in train_drugs and data['drug1_id'] in test_drugs):
                if test2_neg < test2_pos:
                    test_s2.append(data)
                    test2_neg += 1
            else:
                unuse_neg += 1

        print("fold %d train pairs:"%split, len(train_set))
        print("fold %d dev pairs:"%split, len(dev_set))
        print("fold %d test s1 pairs:"%split, len(test_s1))
        print("fold %d test s2 pairs:"%split, len(test_s2))
        print("fold %d unuse_pos:"%split, unuse_pos)
        print("fold %d unuse_neg:"%split, unuse_neg)


        with open("datasets/cross_validation/fold%d/inductive_train.json"%split, "w") as f:
            random.shuffle(train_set)
            for data in train_set:
                f.write(json.dumps(data)+"\n")

        with open("datasets/cross_validation/fold%d/inductive_dev.json"%split, "w") as f:
            random.shuffle(dev_set)
            for data in dev_set:
                f.write(json.dumps(data)+"\n")

        with open("datasets/cross_validation/fold%d/inductive_test_s1.json"%split, "w") as f:
            random.shuffle(test_s1)
            for data in test_s1:
                f.write(json.dumps(data)+"\n")

        with open("datasets/cross_validation/fold%d/inductive_test_s2.json"%split, "w") as f:
            random.shuffle(test_s2)
            for data in test_s2:
                f.write(json.dumps(data)+"\n")

def main():
    collect_positives()
    collect_negatives()
    split_transductive()
    split_inductive()
    prepare_cross_valid_inductive()



if __name__ == '__main__':
    main()