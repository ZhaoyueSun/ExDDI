import os
import openai
import random
import json


class Prompter:
    def __init__(self, train_file, data_src=None, instruction_template=None, query_template=None, random_seed=None):

        if random_seed == None:
            random_seed = random.randint(0, 1000)
        random.seed(random_seed)

        # openai.organization = "org-G1AdYymvZPETTLDmtFln1REi"
        with open('chatgpt/open-ai-key.txt', 'r') as f:
            openai.api_key = f.readline().strip()
        openai.Model.list()

        self.train_file = train_file
        self.data_src = data_src
        self.instruction_template = instruction_template
        self.query_template = query_template

        self.data = []
        self.read_data()

        
    def read_data(self):

        with open(self.train_file) as f:
            for line in f.readlines():
                d = json.loads(line)
                answer = 'Yes. ' if d['label'] else 'No. '
                if self.data_src == 'drugbank':
                    answer += d['drugbank_explanation'] 
                elif self.data_src == 'ddinter':
                    answer += d['ddinter_explanation'] 
                self.data.append({
                    'drug1': d['drug1_smiles'],
                    'drug2': d['drug2_smiles'],
                    'answer': answer
                })
    


        
    def sample_an_id(self, split):
        id_list = self.id_list[split]
        idx = random.randint(0, len(id_list)-1)
        return id_list[idx]
    
    def get_an_instance(self, sample_id):
        return self.data[sample_id]
    
    def get_instruction_prompt(self):

        if os.path.exists(self.instruction_template):
            with open(self.instruction_template, 'r') as f:
                instruction = json.load(f)
                return instruction
        else:
            raise Exception("File does not exists: instruction template!")
        

    def get_an_example_prompt(self, sample_id):
        instance = self.data[sample_id]
        prompt = self.get_query_prompt([instance['drug1'], instance['drug2']])
        answer = instance['answer']
        prompt += [{"role": "assistant", "content": answer}]
        return prompt
    
    def get_query_prompt(self, instance):
        drug1, drug2 = instance
        if os.path.exists(self.query_template):
            with open(self.query_template, 'r') as f:
                prompt = json.load(f)
                prompt[0]['content'] = prompt[0]['content'].replace('<DRUG1_SMILES>', drug1).replace('<DRUG2_SMILES>', drug2)
                return prompt
        else:
            raise Exception("File does not exists: query template!")
    

    def get_chatgpt_response(self, prompts):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=prompts,
        temperature=0
        )

        answer = response['choices'][0]['message']['content']

        return answer
    
    def get_prompts(self, instance, demo_ids):
        prompt = []
        prompt += self.get_instruction_prompt()
        prompt += self.get_demostrations_prompt(demo_ids)
        prompt += self.get_query_prompt(instance)
        return prompt
   
    def get_result(self, instance, demo_ids):
        prompts = self.get_prompts(instance, demo_ids)
        answer = self.get_chatgpt_response(prompts)
        result = {'prompt': prompts,
        'answer': answer}

        return result
    
    def get_demostrations_prompt(self, demo_ids):
        
        prompts = []
        for sample_id in demo_ids:
            prompts += self.get_an_example_prompt(sample_id)

        return prompts


    



    