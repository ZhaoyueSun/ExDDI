"""
Reuse code from: https://github.com/YifanDengWHU/DDIMDL/blob/master/NLPProcess.py
To parse DrugBank DDI descriptions to tuples
"""

import stanza
import numpy as np

#### first run:
# stanza.download('en')


def NLPProcess(druglist, df_interactions):
    def addMechanism(node):
        if int(sonsNum[int(node-1)])==0:
            return
        else:
            for k in sons[node-1]:
                if int(k)==0:
                    break
                if dependency[int(k - 1)].text == drugA[i] or dependency[int(k - 1)].text == drugB[i]:
                    continue
                quene.append(int(k))
                addMechanism(int(k))
        return quene

    nlp = stanza.Pipeline('en')
    event=df_interactions
    mechanism=[]
    action=[]
    drugA=[]
    drugB=[]
    for i in range(len(event)):
        
        doc=nlp(event[i])
        if len(doc.sentences) > 1:
            mechanism.append('None')
            action.append('None')
            drugA.append('None')
            drugB.append('None')
            continue

        try:
            dependency = []
            for j in range(len(doc.sentences[0].words)):
                dependency.append(doc.sentences[0].words[j])
            sons=np.zeros((len(dependency),len(dependency)))
            sonsNum=np.zeros(len(dependency))
            flag=False
            count=0
            drugA_added = False
            drugB_added = False
            action_added = False
            mechanism_added = False
        
            for j in dependency:
                if j.deprel=='root':
                    root=int(j.id)
                    action.append(j.lemma)
                    action_added = True
 
                if j.text in druglist:
                    if count<2:
                        if flag==True:
                            drugB.append(j.text)
                            drugB_added = True
                            count+=1
                        else:
                            drugA.append(j.text)
                            drugA_added = True
                            flag=True
                            count+=1
                sonsNum[j.head-1]+=1
                sons[j.head-1,int(sonsNum[j.head-1]-1)]=int(j.id)

            quene=[]
            for j in range(int(sonsNum[root-1])):
                if dependency[int(sons[root-1,j]-1)].deprel=='obj' or dependency[int(sons[root-1,j]-1)].deprel=='nsubj:pass':
                    quene.append(int(sons[root-1,j]))
                    break
            quene=addMechanism(quene[0])
            quene.sort()
            # mechanism.append(" ".join(dependency[j-1].text for j in quene))
            tmp_mech = " ".join(dependency[j-1].text for j in quene)
            if tmp_mech=="the fluid retaining activities":
                tmp_mech="the fluid"
            if tmp_mech=="atrioventricular blocking ( AV block )":
                tmp_mech='the atrioventricular blocking ( AV block ) activities increase'
            mechanism.append(tmp_mech)
            mechanism_added = True

        except Exception as e:
            # print(f"An error occurred: {e}")
            if mechanism_added:
                mechanism[-1] = 'None'
            else:
                mechanism.append('None')
            if action_added:
                action[-1] = 'None'
            else:
                action.append('None')
            if drugA_added:
                drugA[-1] = 'None'
            else:
                drugA.append('None')
            if drugB_added:
                drugB[-1] = 'None'
            else:
                drugB.append('None')
            continue



    return mechanism,action,drugA,drugB