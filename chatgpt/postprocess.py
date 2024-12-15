import os
import torch
import evaluate
import json


def main():
    gold_file = "datasets/inductive_test_s2.json"
    pred_folder = "output/chatgpt/drugbank/s2_preds"
    out_file = "output/chatgpt/drugbank/s2_result/test_preds.txt"
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    outputs = []

    # read preds
    for sample_id in range(len(os.listdir(pred_folder))):
        pred_file = os.path.join(pred_folder, str(sample_id)+".json")
        with open(pred_file) as f:
            data = json.load(f)
            answer = data['answer']
            if answer.startswith('Yes.'):
                label = "positive"
                exp = answer.replace('Yes.',"",1)
            elif answer.startswith('No.'):
                label = "negative"
                exp = answer.replace('No.',"",1)
            else:
                print("illegal output format:%d"%sample_id)
                label = "negative"
                exp = answer
            outputs.append("%s explanation: %s"%(label, exp.replace("\n", " ").strip()))
            

    with open(out_file, "w") as f:
        f.write("\n".join(outputs))


if __name__ == '__main__':
    main()