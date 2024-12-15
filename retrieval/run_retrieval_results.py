import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from tqdm import tqdm
from retrieval.retriever import Retriever



def get_retrieve_index():
    train_file = "datasets/inductive_train.json"
    # train_file = "datasets/transductive_train.json"

    query_files = ["datasets/inductive_test_s1.json", "datasets/inductive_test_s2.json"]


    model_path = None
    mode = 'drug_smiles_fingerprints'
    # sim_kernels = ['RDKit_Fingerprints', 'Atom_Pairs_Fingerprints', 'Topological_Torsion_Fingerprints', 'Morgan_Fingerprints', 'MACCS_Keys_Fingerprints']
    sim_kernels = ['MACCS_Keys_Fingerprints']
    sim_pooling = None
    
    
    for query_file in query_files:
        
        top_k = 50
        output_folder = os.path.join("datasets/retrieved_idx", "drug_smiles_fingerprints_MACCS_Keys", "top%d"%top_k)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
  
        if 'train' in query_file: # may retrieve the sample itself
            top_k += 1

        retriever = Retriever(train_file, model_path, mode, top_k, device='cuda', encode_batch_size=32, chem_sim_kernel=sim_kernels, sim_pooling=sim_pooling)

        incomple_case = 0
        output_file = os.path.join(output_folder, os.path.basename(query_file))
        with open(output_file, 'w') as fout:
            with open(query_file, 'r') as f:
                for lid, line in tqdm(enumerate(f.readlines()), desc="retrieval process"):
                    data = json.loads(line)
                    if mode == 'random' or 'smiles' in mode:
                        drug1_smiles = data['drug1_smiles']
                        drug2_smiles = data['drug2_smiles']
                        rst_ids = retriever.retrieve_nearest_samples((drug1_smiles, drug2_smiles))
                        if 'train' in query_file:
                            if lid in rst_ids:
                                rst_ids.remove(lid)
                            else:
                                rst_ids.pop(-1)
                    elif 'describ' in mode:
                        drug1_desc = data['drug1_gold_description']
                        drug2_desc = data['drug2_gold_description']
                        rst_ids = retriever.retrieve_nearest_samples((drug1_desc, drug2_desc))
                        if 'train' in query_file:
                            if lid in rst_ids:
                                rst_ids.remove(lid)
                            else:
                                rst_ids.pop(-1)
                    else:
                        raise NotImplementedError

                    if 'train' in query_file:
                        if len(rst_ids) < top_k-1:
                            incomple_case += 1
                    else:
                        if len(rst_ids) < top_k:
                            incomple_case += 1

                    fout.write(json.dumps(rst_ids)+"\n")

        print(query_file)
        print("incomplete case:", incomple_case)



def get_top_retrieve_preds():
    train_file = "datasets/transductive_train.json"
    index_file = "datasets/retrieved_idx/drug_smiles_fingerprints_MACCS_Keys/top50/transductive_test.json"
    output_folder = "output/retrieval/drugbank/trans_result"
    data_src = "drugbank"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder ,"test_preds.txt")

    # read train data
    train_data = []
    with open(train_file) as f:
        for line in f.readlines():
            train_data.append(json.loads(line))

    # read index 
    indexes = []
    with open(index_file) as f:
        for line in f.readlines():
            indexes.append(json.loads(line)[0])

    outputs = []
    for idx in indexes:
        sample = train_data[idx]
        label = "positive" if sample['label'] else "negative"
        exp = sample["ddinter_explanation"] if data_src=='ddinter' else sample['drugbank_explanation']
        exp = exp.replace("\n", " ")
        exp = "%s explanation: %s"%(label, exp)
        outputs.append(exp.strip())

    with open(output_file, "w") as f:
        f.write("\n".join(outputs))

    




if __name__ == '__main__':
    get_top_retrieve_preds()