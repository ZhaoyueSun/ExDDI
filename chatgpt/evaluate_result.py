import os
import torch
import evaluate
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    # compute bert score
    bertscore = evaluate.load("bertscore")
    bertscore_result = bertscore.compute(predictions=preds, references=refs, lang="en")
    results['bertscore_p'] = sum(bertscore_result['precision']) / len(bertscore_result['precision'])
    results['bertscore_r'] = sum(bertscore_result['recall']) / len(bertscore_result['recall'])
    results['bertscore_f1'] = sum(bertscore_result['f1']) / len(bertscore_result['f1'])

    results = {k: round(v, 4) for k, v in results.items()}
    return results



def main():
    gold_file = "data/dataset/inductive_test_s1.json"
    pred_folder = "chatgpt/output/drug_describ_bm25_top5/test_s1"

    preds = []
    golds = []
    pos_preds = []
    pos_golds = []
    neg_preds = []
    neg_golds = []
    gold_labels = []
    pred_labels = []

    # read preds
    for sample_id in range(len(os.listdir(pred_folder))):
        pred_file = os.path.join(pred_folder, str(sample_id)+".json")
        with open(pred_file) as f:
            data = json.load(f)
            answer = data['answer']
            if answer.startswith('Yes.'):
                pred_labels.append(1)
                exp = answer.replace('Yes.',"",1)
            elif answer.startswith('No.'):
                pred_labels.append(0)
                exp = answer.replace('No.',"",1)
            else:
                print("illegal output format:%d"%sample_id)
                pred_labels.append(0)
                exp = answer
            preds.append(exp)
            

    # read gold data
    with open(gold_file) as f:
        for lid, line in enumerate(f.readlines()):
            data = json.loads(line)
            gold = data['explanation']
            golds.append(gold)
            pred = preds[lid]

            label = data['label']
            gold_labels.append(1) if label else gold_labels.append(0)
            if label:
                pos_golds.append(gold)
                pos_preds.append(pred)
            else:
                neg_golds.append(gold)
                neg_preds.append(pred)

    
    # classification scores
    accuracy = accuracy_score(gold_labels, pred_labels)
    print("Accuracy:", accuracy)

    precision = precision_score(gold_labels, pred_labels)
    print("Precision:", precision)

    recall = recall_score(gold_labels, pred_labels)
    print("Recall:", recall)

    f1 = f1_score(gold_labels, pred_labels)
    print("F1 Score:", f1)

    # overall scores
    scores = compute_metric(golds, preds)
    print("######## Overall Score #########")
    for k, v in scores.items():
        print("%s: %.4f"%(k, v))

    # pos scores 
    pos_scores = compute_metric(pos_golds, pos_preds)
    print("######## Positive Case Score #########")
    for k, v in pos_scores.items():
        print("%s: %.4f"%(k, v))

    # neg scores 
    neg_scores = compute_metric(neg_golds, neg_preds)
    print("######## Negative Case Score #########")
    for k, v in neg_scores.items():
        print("%s: %.4f"%(k, v))


if __name__ == '__main__':
    main()