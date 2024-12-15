import evaluate
import json
import csv
import Levenshtein
import os
from collections import OrderedDict, defaultdict
from prepare_data.NLPProcess import NLPProcess
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def read_ddinter_templates():
    templates = {}
    temp_path = "datasets/ddinter_ddi_mechanisms.csv"
    with open(temp_path, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            templates[line[2]] = int(line[0])
    
    return templates

def read_drugbank_templates():
    templates = {}
    temp_path = "datasets/drugbank_ddi_mechanisms.csv"
    with open(temp_path, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            mechanism = "_".join([line[1], line[2]])
            templates[mechanism] = int(line[0])
    
    return templates

def find_nearest_drugbank_template(query, templates):
    temp_list = [tuple(x.split('_')) for x in templates.keys()]
    query_m, query_a = query.split('_')
    scores = [Levenshtein.ratio(query_m, temp[0]) if query_a == temp[1] else 0 for temp in temp_list]
    sim = temp_list[scores.index(max(scores))]
    return templates["_".join(sim)]


def read_ddinter_drugbank_template_map():
    mapping = {}
    with open("datasets/ddinter_to_drugbank_mechanism_map.csv", newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            mapping[int(line[0])] = int(line[1])
    return mapping

def postprocess_text(x_str):

    to_remove_token_list = ['<pad>', '</s>']
    for to_remove_token in to_remove_token_list:
        x_str = x_str.replace(to_remove_token, '')
    x_str = x_str.strip()

    if x_str.startswith('positive'):
        x_str = x_str.replace('positive', '', 1).strip()
        l_pred = True
    elif x_str.startswith('negative'):
        x_str = x_str.replace('negative', '', 1).strip()
        l_pred = False
    else:
        l_pred = False

    if x_str.startswith("explanation:"):
        x_str = x_str.replace("explanation:", "", 1).strip()
    
    return x_str, l_pred

def compute_metric(refs, preds):
    """
    refs: list of reference text.
    preds: list of predicted text.
    """
    results = {}
    # compute BLEU score
    bleu_metric = evaluate.load("bleu")
    bleu_result = bleu_metric.compute(predictions=preds, references=refs)
    results['bleu'] = bleu_result['bleu']

    # compute rouge score
    rouge_metric = evaluate.load('rouge')
    rouge_result = rouge_metric.compute(predictions=preds, references=refs)
    results.update(rouge_result)

    # # compute bert score
    # bertscore = evaluate.load("bertscore")
    # bertscore_result = bertscore.compute(predictions=preds, references=refs, lang="en")
    # results['bertscore_p'] = sum(bertscore_result['precision']) / len(bertscore_result['precision'])
    # results['bertscore_r'] = sum(bertscore_result['recall']) / len(bertscore_result['recall'])
    # results['bertscore_f1'] = sum(bertscore_result['f1']) / len(bertscore_result['f1'])

    results = {k: round(v, 4) for k, v in results.items()}
    return results



def main():
    # fold_id = 1
    for fold_id in [2,3,4,5]:
        data_src = "ddinter"
        model_name = "mult_task_training_base_ddinter_inductive_5e-4"
        rst_folder = "s2_result_v2"
        test_file = "inductive_test_s2.json"


        gold_file = "datasets/cross_validation/fold%d/%s"%(fold_id, test_file)
        pred_file = "output/%s_fold%d/%s/test_preds.txt"%(model_name, fold_id, rst_folder)
        result_output = "output/%s_fold%d/%s/output_metrics.json"%(model_name, fold_id, rst_folder)
        

        raw_preds = []
        preds = []
        golds = []
        pos_preds = []
        pos_golds = []
        neg_preds = []
        neg_golds = []
        exp_label_gold = []
        exp_label_pred = []
        label_gold = []
        label_pred = []


        if data_src == 'drugbank':
            templates = read_drugbank_templates()
        else:
            templates = read_ddinter_templates()
        temp_list = list(templates.keys())

        # read pred data
        if data_src == 'drugbank':
            pred_exps = OrderedDict()
            with open(pred_file) as f:
                for lid, line in enumerate(f.readlines()):
                    pred, lpred = postprocess_text(line)
                    if lpred:
                        pred_exps[lid] = pred
                    raw_preds.append(line)
            mechanisms ,actions ,_ ,_ = NLPProcess(["DRUG1", "DRUG2"], list(pred_exps.values()))
            for eid, lid in enumerate(pred_exps.keys()):
                m = mechanisms[eid].lower()
                a = actions[eid].lower()
                pred_exps[lid] = "_".join([m, a])
            
        else:
            with open(pred_file) as f:
                for line in f.readlines():
                    raw_preds.append(line)

            temp_mapping = read_ddinter_drugbank_template_map()

        # read gold data
        with open(gold_file) as f:
            for lid, line in enumerate(f.readlines()):
                data = json.loads(line)

                if data_src == 'ddinter':
                    gold = data['ddinter_explanation']
                elif data_src == 'drugbank':
                    gold = data['drugbank_explanation']
                else:
                    raise NotImplementedError

                golds.append(gold)
                label = data['label']
                label_gold.append(label)

                pred = raw_preds[lid]
                pred, lpred = postprocess_text(pred)
                preds.append(pred)
                label_pred.append(lpred)
                
                if label:
                    pos_golds.append(gold)
                    pos_preds.append(pred)
                else:
                    neg_golds.append(gold)
                    neg_preds.append(pred)

                # exp matching eval
                if data_src == 'ddinter':
                    exp_label_gold.append(temp_mapping[int(data['ddinter_exp_id'])])
                    explanation = pred
                    if not lpred:
                        exp_label_pred.append(0) 
                    elif explanation in templates:
                        exp_id = templates[explanation]
                        exp_label_pred.append(temp_mapping[exp_id])
                    else:
                        scores = [Levenshtein.ratio(explanation, sentence) for sentence in temp_list]
                        sim = temp_list[scores.index(max(scores))]
                        exp_id = templates[sim]
                        exp_label_pred.append(temp_mapping[exp_id])
                else:  # drugbank
                    exp_label_gold.append(int(data['drugbank_exp_id']))
                    
                    if not lpred:
                        exp_label_pred.append(0)
                    else:
                        pred_mechanism = pred_exps[lid] 
                        # print(pred_mechanism)
                        if pred_mechanism == 'none_none':
                            exp_label_pred.append(0)
                            print("inconsistent prediction: %s"%raw_preds[lid])
                        elif pred_mechanism in templates:
                            exp_label_pred.append(templates[pred_mechanism])
                        else:
                            exp_label_pred.append(find_nearest_drugbank_template(pred_mechanism, templates))

        with open(result_output, "w") as f:

            final_score = defaultdict(dict)

            # binary classification
            cls_metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])
            cls_result = cls_metric.compute(predictions=label_pred, references=label_gold)
            # f.write("######## Classification Score #########\n")
            for k in ["accuracy", "f1", "precision", "recall"]:
                # f.write("%s: %.2f\n"%(k, 100*cls_result[k]))
                final_score['binary_cls'][k] = "%.4f"%cls_result[k]

            # explanation classification
            accuracy = evaluate.load("accuracy")
            acc_result = accuracy.compute(predictions=exp_label_pred, references=exp_label_gold)
            recall = evaluate.load("recall")
            recall_result = recall.compute(predictions=exp_label_pred, references=exp_label_gold, average="macro")
            precision = evaluate.load("precision")
            precision_result = precision.compute(predictions=exp_label_pred, references=exp_label_gold, average="macro")
            f1 = evaluate.load("f1")
            f1_result = f1.compute(predictions=exp_label_pred, references=exp_label_gold, average="macro")

            # f.write("######## Explanation Matching Score #########\n")
            # f.write("%s: %.2f\n"%("accuracy", 100*acc_result["accuracy"]))
            # f.write("%s: %.2f\n"%("f1", 100*f1_reulst["f1"]))
            # f.write("%s: %.2f\n"%("precision", 100*precision_result["precision"]))
            # f.write("%s: %.2f\n"%("recall", 100*recall_result["recall"]))
            final_score['mult_cls']["accuracy"] = "%.4f"%acc_result["accuracy"]
            final_score['mult_cls']["f1"] = "%.4f"%f1_result["f1"]
            final_score['mult_cls']["precision"] = "%.4f"%precision_result["precision"]
            final_score['mult_cls']["recall"] = "%.4f"%recall_result["recall"]

            # overall scores
            scores = compute_metric(golds, preds)
            # f.write("######## Overall Score #########\n")
            for k, v in scores.items():
                # f.write("%s: %.2f\n"%(k, 100*v))
                final_score['overall_score'][k] = "%.4f"%v

            # pos scores 
            pos_scores = compute_metric(pos_golds, pos_preds)
            # f.write("######## Positive Case Score #########\n")
            for k, v in pos_scores.items():
                # f.write("%s: %.2f\n"%(k, 100*v))
                final_score['pos_score'][k] = "%.4f"%v

            # neg scores 
            neg_scores = compute_metric(neg_golds, neg_preds)
            # f.write("######## Negative Case Score #########\n")
            for k, v in neg_scores.items():
                # f.write("%s: %.2f\n"%(k, 100*v))
                final_score['neg_score'][k] = "%.4f"%v

            output_str = json.dumps(final_score, indent=4)
            f.write(output_str)


if __name__ == '__main__':
    main()