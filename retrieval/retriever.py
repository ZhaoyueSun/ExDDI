import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import random
import json
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from transformers import AutoTokenizer, AutoConfig, T5EncoderModel
# from models.t5_encoder.modeling_t5_encoder_snnl import T5EncoderForSnnlEmbedding
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import spacy
import torch
import gzip
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Draw
import math
# from rdkit.Avalon import pyAvalonTools

class Retriever:
    def __init__(self, train_file, model_path=None, retrieve_mode='random', top_k=1, device='cpu', encode_batch_size=1, random_seed=None, chem_sim_kernel=[], sim_pooling=None):
        """
        train_file: the path to the data file storing the training examples;
        model_path: the path of model to compute representations;
        retrieve_mode: random, ddi_smiles_dense, drug_smiles_dense, drug_describ_dense, drug_describ_bm25, ddi_smiles_molt5, drug_smiles_fingerprints;
        top_k: return top_k nearest DDI pairs;
        device: 'cpu' or 'cuda' will be used for computing representations
        random_seed: set random seed;
        chem_sim_kernel: name of chemical similarity metric to choose, available: RDKit_Fingerprints, Atom_Pairs_Fingerprints, Topological_Torsions_Fingerprints, Morgan_Fingerprints, MACCS_Keys_Fingerprints;
        sim_pooling: pooling strategy for combining chemical similarity
        """

        if random_seed == None:
            random_seed = random.randint(0, 1000)
        random.seed(random_seed)

        self.train_file = train_file
        self.model_path = model_path
        self.retrieve_mode = retrieve_mode
        self.top_k = top_k
        self.device = device
        self.chem_sim_kernel = chem_sim_kernel
        self.sim_pooling = sim_pooling

        if retrieve_mode in ['random', 'ddi_smiles_dense', 'ddi_smiles_molt5']:
            self.retrieve_by_drug = False
        elif retrieve_mode in ['drug_smiles_dense', 'drug_describ_dense', 'drug_describ_bm25', 'drug_smiles_fingerprints']:
            self.retrieve_by_drug = True
        else:
            raise NotImplementedError

        self.train_ddis = [] # when self.retrieve_by_drug is False
        self.train_drugs =  OrderedDict() # DDInterID to drug Representation
        self.drug2train_id = defaultdict(list)
        self.read_data()

        print("start encoding training data...")

        if self.retrieve_mode in ['ddi_smiles_dense', 'ddi_smiles_molt5']:
            config = AutoConfig.from_pretrained(model_path)
            config.add_prediction_loss = False
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = T5EncoderForSnnlEmbedding.from_pretrained(model_path, config=config)
            if self.device == 'cuda':
                self.model = self.model.cuda()
            self.model.eval()

            if self.retrieve_mode == 'ddi_smiles_dense':
                inputs = []
                for drug1, drug2 in self.train_ddis:
                    inputs.append("%s %s %s"%(drug1, self.tokenizer.sep_token, drug2))
            else:
                inputs = []
                for drug1, drug2 in self.train_ddis:
                    inputs.append("DRUG1 %s ; DRUG2 %s"%(drug1, drug2))
            
            # self.embeddings = self.encode_embedding(inputs, use_top_layer=True, batch_size = encode_batch_size)
            self.embeddings = self.encode_embedding(inputs, use_top_layer=False, batch_size = encode_batch_size)

        elif self.retrieve_mode in ['drug_smiles_dense', 'drug_describ_dense']:
            config = AutoConfig.from_pretrained(model_path)
            config.add_prediction_loss = False
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = T5EncoderForSnnlEmbedding.from_pretrained(model_path, config=config)
            if self.device == 'cuda':
                self.model = self.model.cuda()
            self.model.eval()

            inputs = []
            for drug in list(self.train_drugs.values()):
                inputs.append(drug)
            
            self.embeddings = self.encode_embedding(inputs, use_top_layer=False, batch_size = encode_batch_size)

            print(len(self.embeddings))

        elif self.retrieve_mode == 'drug_describ_bm25':
            tokenized_descriptions = []
            for description in list(self.train_drugs.values()):
                tokenized_descriptions.append(description.split(" "))
            self.bm25 = BM25Okapi(tokenized_descriptions)

        elif self.retrieve_mode == 'drug_smiles_fingerprints': 
            self.fingerprints = defaultdict(list)
            for sim_kernel in self.chem_sim_kernel:
                break_molcules = 0
                if sim_kernel == 'RDKit_Fingerprints':
                    fpgen = AllChem.GetRDKitFPGenerator()
                elif sim_kernel == 'Atom_Pairs_Fingerprints':
                    fpgen = AllChem.GetAtomPairGenerator()
                elif sim_kernel == 'Topological_Torsion_Fingerprints':
                    fpgen = AllChem.GetTopologicalTorsionGenerator()
                elif sim_kernel == 'Morgan_Fingerprints':
                    fpgen = AllChem.GetMorganGenerator(radius=2)
                elif sim_kernel == 'MACCS_Keys_Fingerprints':
                    pass  # not support generator, use another interface

                for drug_id, drug in self.train_drugs.items():
                    mol = Chem.MolFromSmiles(drug)
                    if not mol:
                        break_molcules += 1
                        self.fingerprints[sim_kernel].append(None)
                        continue

                    if sim_kernel in ['RDKit_Fingerprints']:
                        fp = fpgen.GetFingerprint(mol)
                    elif sim_kernel in ['Atom_Pairs_Fingerprints', 'Topological_Torsion_Fingerprints', 'Morgan_Fingerprints']:
                        fp = fpgen.GetSparseCountFingerprint(mol)
                    elif sim_kernel == 'MACCS_Keys_Fingerprints':
                        fp = MACCSkeys.GenMACCSKeys(mol)
                    
                    self.fingerprints[sim_kernel].append(fp)
                    
                print("break molucles among %d training drugs, %s: %d"%(len(self.fingerprints[sim_kernel]), sim_kernel, break_molcules))

        print("finish encoding training data...")

        
    def encode_embedding(self, inputs, use_top_layer=False, batch_size=1):
        """
        inputs:  input text,
        use_top_layer: if true, use contrastive learning mapped layer embeddings, otherwise use encoder embeddings
        """
        num_samples = len(inputs)
        embeddings = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_inputs = inputs[i:i+batch_size]
                batch_inputs = self.tokenizer(batch_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
                if self.device == 'cuda':
                    input_ids = batch_inputs.input_ids.cuda()
                    attention_mask = batch_inputs.attention_mask.cuda()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                if use_top_layer: 
                    outputs = outputs.mapped_reps
                else:
                    outputs = outputs.encoder_reps
                embeddings.append(outputs) 

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings


    def read_data(self):
        with open(self.train_file) as f:
            for lid, line in enumerate(f.readlines()):
                data = json.loads(line)
                drug1_smiles = data['drug1_smiles']
                drug2_smiles = data['drug2_smiles']
                self.train_ddis.append((drug1_smiles, drug2_smiles))

                if self.retrieve_by_drug:
                    drug1_id = data['drug1_id']
                    drug2_id = data['drug2_id']

                    if 'drug_smiles' in self.retrieve_mode:
                        if drug1_id not in self.train_drugs:
                            self.train_drugs[drug1_id] = drug1_smiles
                        if drug2_id not in self.train_drugs:
                            self.train_drugs[drug2_id] = drug2_smiles
                    elif 'drug_describ' in self.retrieve_mode:
                        if drug1_id not in self.train_drugs:
                            self.train_drugs[drug1_id] = data['drug1_gold_description']
                        if drug2_id not in self.train_drugs:
                            self.train_drugs[drug2_id] = data['drug2_gold_description']

                    self.drug2train_id[drug1_id].append(lid)
                    self.drug2train_id[drug2_id].append(lid)


    
    def retrieve_nearest_samples(self, query_ddi, return_smiles=False):
        """
        query_ddi: tuple = (drug1_rep, drug2_rep)
        return_reps: whether to return smiles of ddi pairs

        return: by default return a list of train_ids, if return_smiles is True, return tuple(list(train_ids), list(tuple(drug1_smiles, drug2_smiles)))

        """
        if self.retrieve_mode == 'random':
            select_ids = random.sample(range(len(self.train_ddis)), self.top_k)

        elif self.retrieve_mode == 'ddi_smiles_molt5':
            select_ids = []
            top_k = min(self.top_k, len(self.embeddings))
            drug1, drug2 = query_ddi
            query = "DRUG1 %s ; DRUG2 %s"%(drug1, drug2)

            query_embedding = self.encode_embedding([query], use_top_layer=False)[0]
            cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            select_ids = [x for x in top_results.indices.cpu().tolist()]

        elif self.retrieve_mode == 'ddi_smiles_dense':
            select_ids = []
            top_k = min(self.top_k, len(self.embeddings))
            drug1, drug2 = query_ddi
            query = "%s %s %s"%(drug1, self.tokenizer.sep_token, drug2)
            # query_embedding = self.encode_embedding([query], use_top_layer=True)[0]
            query_embedding = self.encode_embedding([query], use_top_layer=False)[0]

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            select_ids = [x for x in top_results.indices.cpu().tolist()]

            # distances = torch.cdist(query_embedding.unsqueeze(0), self.embeddings)
            # top_results = torch.topk(distances.squeeze(), k=top_k, largest=False)
            # select_ids = [x for x in top_results.indices.cpu().tolist()]

        elif self.retrieve_mode in ['drug_smiles_dense', 'drug_describ_dense']:
            top_k = 50
            drug_ids = list(self.train_drugs.keys())
            drug1, drug2 = query_ddi
            drug1_embed = self.encode_embedding([drug1], use_top_layer=False)[0]
            drug2_embed = self.encode_embedding([drug2], use_top_layer=False)[0]
            drug1_scores = util.cos_sim(drug1_embed, self.embeddings)[0]
            drug2_scores = util.cos_sim(drug2_embed, self.embeddings)[0]
            drug1_top_rst = torch.topk(drug1_scores, k=top_k)
            drug2_top_rst = torch.topk(drug2_scores, k=top_k)

            cand_ids = []
            for score1, idx1 in zip(drug1_top_rst[0], drug1_top_rst[1]):
                for score2, idx2 in zip(drug2_top_rst[0], drug2_top_rst[1]):
                    if idx1 == idx2:
                        continue
                    drug1 = drug_ids[idx1]
                    drug2 = drug_ids[idx2]
                    drug1_pairs = self.drug2train_id[drug1]
                    drug2_pairs = self.drug2train_id[drug2]
                    inter = set(drug1_pairs) & set(drug2_pairs)
                    if inter:
                        assert (len(inter)==1)
                        trainid = list(inter)[0]
                        cand_ids.append((trainid, score1*score2))

            cand_ids.sort(key=lambda x:x[1], reverse=True)
            cand_ids = cand_ids[:self.top_k]
            select_ids = [x[0] for x in cand_ids]

        
        elif self.retrieve_mode == 'drug_describ_bm25':
            top_k = 50
            drug_ids = list(self.train_drugs.keys())
            drug1_desc, drug2_desc = query_ddi
            drug1_desc = drug1_desc.split(" ")
            drug2_desc = drug2_desc.split(" ")
            drug1_scores = self.bm25.get_scores(drug1_desc)
            drug2_scores = self.bm25.get_scores(drug2_desc)
            drug1_top_rst = torch.topk(torch.tensor(drug1_scores), top_k)
            drug2_top_rst = torch.topk(torch.tensor(drug2_scores), top_k)
                
            cand_ids = []
            for score1, idx1 in zip(drug1_top_rst[0], drug1_top_rst[1]):
                for score2, idx2 in zip(drug2_top_rst[0], drug2_top_rst[1]):
                    if idx1 == idx2:
                        continue
                    drug1 = drug_ids[idx1]
                    drug2 = drug_ids[idx2]
                    drug1_pairs = self.drug2train_id[drug1]
                    drug2_pairs = self.drug2train_id[drug2]
                    inter = set(drug1_pairs) & set(drug2_pairs)
                    if inter:
                        assert (len(inter)==1)
                        trainid = list(inter)[0]
                        cand_ids.append((trainid, score1*score2))

            cand_ids.sort(key=lambda x:x[1], reverse=True)
            cand_ids = cand_ids[:self.top_k]
            select_ids = [x[0] for x in cand_ids]
        
        elif self.retrieve_mode == 'drug_smiles_fingerprints':
            top_k = 50
            drug_ids = list(self.train_drugs.keys())
            drug1_smiles, drug2_smiles = query_ddi
            drug1_mol = Chem.MolFromSmiles(drug1_smiles)
            drug2_mol = Chem.MolFromSmiles(drug2_smiles)

            if not drug1_mol or not drug2_mol: # 不符合格式的molcule，随机采样
                select_ids = random.sample(range(len(self.train_ddis)), top_k)
                if not drug1_mol:
                    print("break query molecule:", drug1_smiles)
                else:
                    print("break query molecule:", drug2_smiles)
            else:
                drug1_scores = defaultdict(list)
                drug2_scores = defaultdict(list)
                for sim_kernel in self.chem_sim_kernel:
                    if sim_kernel == 'RDKit_Fingerprints':
                        fpgen = AllChem.GetRDKitFPGenerator()
                    elif sim_kernel == 'Atom_Pairs_Fingerprints':
                        fpgen = AllChem.GetAtomPairGenerator()
                    elif sim_kernel == 'Topological_Torsion_Fingerprints':
                        fpgen = AllChem.GetTopologicalTorsionGenerator()
                    elif sim_kernel == 'Morgan_Fingerprints':
                        fpgen = AllChem.GetMorganGenerator(radius=2)
                    elif sim_kernel == 'MACCS_Keys_Fingerprints':
                        pass  # not support generator, use another interface

                    if sim_kernel in ['RDKit_Fingerprints']:
                        drug1_fp = fpgen.GetFingerprint(drug1_mol)
                        drug2_fp = fpgen.GetFingerprint(drug2_mol)
                        drug1_scores[sim_kernel] = [DataStructs.TanimotoSimilarity(drug1_fp, fp) if fp else 0 for fp in self.fingerprints[sim_kernel]]
                        drug2_scores[sim_kernel] = [DataStructs.TanimotoSimilarity(drug2_fp, fp) if fp else 0 for fp in self.fingerprints[sim_kernel]]
                    elif sim_kernel in ['Atom_Pairs_Fingerprints', 'Topological_Torsion_Fingerprints', 'Morgan_Fingerprints']:
                        drug1_fp = fpgen.GetSparseCountFingerprint(drug1_mol)
                        drug2_fp = fpgen.GetSparseCountFingerprint(drug2_mol)
                        drug1_scores[sim_kernel] = [DataStructs.DiceSimilarity(drug1_fp, fp) if fp else 0 for fp in self.fingerprints[sim_kernel]]
                        drug2_scores[sim_kernel] = [DataStructs.DiceSimilarity(drug2_fp, fp) if fp else 0 for fp in self.fingerprints[sim_kernel]]
                    elif sim_kernel == 'MACCS_Keys_Fingerprints':
                        drug1_fp = MACCSkeys.GenMACCSKeys(drug1_mol)
                        drug2_fp = MACCSkeys.GenMACCSKeys(drug2_mol)
                        drug1_scores[sim_kernel] = [DataStructs.TanimotoSimilarity(drug1_fp, fp) if fp else 0 for fp in self.fingerprints[sim_kernel]]
                        drug2_scores[sim_kernel] = [DataStructs.TanimotoSimilarity(drug2_fp, fp) if fp else 0 for fp in self.fingerprints[sim_kernel]]

                if self.sim_pooling == 'Max':
                    pooled_drug1_scores = [max(x) for x in zip(*drug1_scores.values())]
                    pooled_drug2_scores = [max(x) for x in zip(*drug2_scores.values())]
                elif self.sim_pooling == 'Mean':
                    pooled_drug1_scores = [sum(x)/len(x) for x in zip(*drug1_scores.values())]
                    pooled_drug2_scores = [sum(x)/len(x) for x in zip(*drug2_scores.values())]
                elif self.sim_pooling == 'Geometric_Mean':
                    pooled_drug1_scores = [math.pow(math.prod(x), 1/len(x)) for x in zip(*drug1_scores.values())]
                    pooled_drug2_scores = [math.pow(math.prod(x), 1/len(x)) for x in zip(*drug2_scores.values())]
                else:
                    pooled_drug1_scores = drug1_scores[self.chem_sim_kernel[0]]
                    pooled_drug2_scores = drug2_scores[self.chem_sim_kernel[0]]

                drug1_top_rst = torch.topk(torch.tensor(pooled_drug1_scores), top_k)
                drug2_top_rst = torch.topk(torch.tensor(pooled_drug2_scores), top_k)
                    
                cand_ids = []
                for score1, idx1 in zip(drug1_top_rst[0], drug1_top_rst[1]):
                    for score2, idx2 in zip(drug2_top_rst[0], drug2_top_rst[1]):
                        if idx1 == idx2:
                            continue
                        drug1 = drug_ids[idx1]
                        drug2 = drug_ids[idx2]
                        drug1_pairs = self.drug2train_id[drug1]
                        drug2_pairs = self.drug2train_id[drug2]
                        inter = set(drug1_pairs) & set(drug2_pairs)
                        if inter:
                            assert (len(inter)==1)
                            trainid = list(inter)[0]
                            cand_ids.append((trainid, score1*score2))

                cand_ids.sort(key=lambda x:x[1], reverse=True)
                cand_ids = cand_ids[:self.top_k]
                select_ids = [x[0] for x in cand_ids]

        else:
            raise NotImplementedError

        if not return_smiles:
            return select_ids
        else:
            select_reps = [self.train_ddis[i] for i in select_ids]
            return select_ids, select_reps




    