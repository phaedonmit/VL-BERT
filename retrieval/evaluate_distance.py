######
# Author: Faidon Mitzalis
# Date: June 2020
# Comments: Use results of testing to evaluate distance between [CLS] embeddings
######

import json
import torch
import operator 
import numpy as np

# filepath_en = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/pretrain_prec/002_prec_LR1e6_multi30k/distance_002_prec_LR1e6_multi30k_cls_English_retrieval_translation_test2015.json"
filepath_en = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/pretrain_prec/003_prec_LR1e6_taskB_multi30k_5x/003_prec_LR1e6_taskB_multi30k_5x_cls_output_English_retrieval_translation_test2015.json"
# filepath_de = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/pretrain_prec/002_prec_LR1e6_multi30k/distance_002_prec_LR1e6_multi30k_cls_German_retrieval_translation_test2015.json"
filepath_de = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/pretrain_prec/003_prec_LR1e6_taskB_multi30k_5x/003_prec_LR1e6_taskB_multi30k_5x_cls_output_German_retrieval_translation_test2015.json"

# Step 1 - Read json files with outputs
with open(filepath_en) as json_file:
    data_en = json.load(json_file)
with open(filepath_de) as json_file:
    data_de = json.load(json_file)

assert(len(data_en)==len(data_de))

# Step 2 - Convert to numpy arrays
outputs_en = np.zeros((len(data_en), len(data_en[0]['logit'])))
outputs_de = np.zeros((len(data_en), len(data_en[0]['logit'])))

for ind, p in enumerate(data_en):
    outputs_en[ind] = np.array(p['logit'])
for ind, p in enumerate(data_de):
    outputs_de[ind] = np.array(p['logit'])


# Step 3 - Compute cosine similarities
x = outputs_en
y = outputs_de
cos = (x@y.T)/np.outer(np.linalg.norm(x, axis=1),np.linalg.norm(y, axis=1))


# Step 4 - Compute Ranks
print('***************************************')
print('***** German Caption retrieval    *****')
print('***************************************')
print('Total captions assessed: ', cos.shape[0])
print('******')
cos_sorted = np.argsort(cos, axis=1)[:,::-1]
answers = np.arange(cos.shape[0])
for rank in [1, 5,10]:
    total_correct = 0
    for position in range(rank):
        total_correct += np.sum(cos_sorted[:,position]==answers)
    print(f'Total correct for R{rank}: ', total_correct)
    print(f'Accuracy for R{rank}%: {total_correct/cos.shape[0]*100:.2f}%')
    print('*****')

# Step 4 - Compute Ranks
print('****************************************')
print('***** English Caption retrieval    *****')
print('****************************************')
print('Total captions assessed: ', cos.shape[0])
print('******')
cos = cos.T
cos_sorted = np.argsort(cos, axis=1)[:,::-1]
answers = np.arange(cos.shape[0])
for rank in [1, 5,10]:
    total_correct = 0
    for position in range(rank):
        total_correct += np.sum(cos_sorted[:,position]==answers)
    print(f'Total correct for R{rank}: ', total_correct)
    print(f'Accuracy for R{rank}%: {total_correct/cos.shape[0]*100:.2f}%')
    print('*****')    

