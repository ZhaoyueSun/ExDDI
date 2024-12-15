import os
import json
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
import tqdm
from chatgpt.prompter import Prompter

def main():
    train_file = "datasets/inductive_train.json"
    rtv_idx_file = "datasets/retrieved_idx/drug_smiles_fingerprints_MACCS_Keys/top50/inductive_test_s2.json"
    query_file = "datasets/inductive_test_s2.json"
    demo_num = 5
    data_src = "ddinter"
    instruction_template = "chatgpt/instruction.json"
    query_template = "chatgpt/query.json"

    prompter = Prompter(train_file, data_src=data_src, instruction_template=instruction_template, query_template=query_template, random_seed=42)

    output_folder = "output/chatgpt/%s/s2_preds/"%data_src
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log_file = os.path.join(output_folder, 'fail_cases.json')

    # read fail cases
    with open(log_file) as f:
        fail_cases = json.load(f)

    # read retrieve idx
    indexes = []
    with open(rtv_idx_file) as f:
        for line in f.readlines():
            indexes.append(json.loads(line)[:demo_num])

    query_list = []
    with open(query_file) as f:
        for line in f.readlines():
            data = json.loads(line)
            query_list.append([data['drug1_smiles'], data['drug2_smiles']])


    
    new_fail_cases = []
    count = 0
    for sample_id, query in tqdm.tqdm(enumerate(query_list)):
        count += 1
        if os.path.exists(os.path.join(output_folder, str(sample_id)+".json")):
            continue
        # if count > 3: break
        try:
            result = prompter.get_result(query, indexes[sample_id])
        except:
            new_fail_cases.append(sample_id)
            with open(log_file, 'w') as f:
                json.dump(new_fail_cases, f)
            print("fail case: %s"%sample_id)
            continue

        with open(os.path.join(output_folder, str(sample_id)+".json"), 'w') as f:
            json.dump(result, f)
            

if __name__ == '__main__':
    main()