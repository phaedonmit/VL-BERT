######
# Author: Faidon Mitzalis
# Date: June 2020
# Comments: Use results of testing to evaluate performance of image retrieval task
######

import json
import torch
import operator 

# filepath = "/experiments/faidon/VL-BERT/checkpoints/itm_evaluation_retrieval_test2015.json"
# filepath = "/experiments/faidon/VL-BERT/checkpoints/imt_model19_prec_5captions_LR1e6_all.json"
filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/translation_retrieval/001_prec_translation_retrieval_with_vision_19model_retrieval_translation_test2015.json"

with open(filepath) as json_file:
    data = json.load(json_file)
    ranks = [1, 5, 10]
    correct = [0, 0, 0]
    total=0
    captions_en_dict = {}
    captions_de_dict = {}

    #******************************************************
    # Step 1: Read json to dictionaries
    
    # json file to nested dictionary (each caption with all images)
    for p in data:
        if p['caption_en_index'] in captions_en_dict.keys():
            captions_en_dict[p['caption_en_index']][p['caption_de_index']]= ((torch.tensor(p['logit'][0])).item())
        else:
            captions_en_dict[p['caption_en_index']] = {p['caption_de_index']: (torch.tensor(p['logit'][0])).item()}
    # # json file to nested dictionary (each image with all caption)
        if p['caption_de_index'] in captions_de_dict.keys():
            captions_de_dict[p['caption_de_index']][p['caption_en_index']]= ((torch.tensor(p['logit'][0])).item())
        else:
            captions_de_dict[p['caption_de_index']] = {p['caption_en_index']: (torch.tensor(p['logit'][0])).item()}

    #******************************************************
    # Step 2: Get ranks image retrieval
    # get r1, r5 and r10 for each capion
    for index, value in captions_en_dict.items():
        # print('-------')
        # print('Top candiate is: ', (max(value.items(), key=operator.itemgetter(1))))
        # print('Actual image id is: ', index)
        if (max(value.items(), key=operator.itemgetter(1)))[0] == index:
            correct[0] += 1
        # Get top 5
        if index in dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[1]]).keys():
            correct[1] += 1
        if index in dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[2]]).keys():
            correct[2] += 1    
        total += 1
    print('******************************')
    print('***** German Caption retrieval    *****')
    print('******************************')
    print('Total captions assessed: ', total)
    print('******')
    print(f'Correct rank {ranks[0]}: ', correct[0])
    print(f'Correct rank {ranks[1]}: ', correct[1])
    print(f'Correct rank {ranks[2]}: ', correct[2])
    print('******')
    print(f"The total percentage rank {ranks[0]} is {correct[0]/total*100:.2f}%")
    print(f"The total percentage rank {ranks[1]} is {correct[1]/total*100:.2f}%")
    print(f"The total percentage rank {ranks[2]} is {correct[2]/total*100:.2f}%")



    #******************************************************
    # Step 3: Get ranks caption retrieval
    # get r1, r5 and r10 for each capion
    correct = [0, 0, 0]
    total=0    
    for index, value in captions_de_dict.items():
        # print('-------')
        # print('Top candiate is: ', (max(value.items(), key=operator.itemgetter(1))))
        # print('Actual image id is: ', index)
        if (max(value.items(), key=operator.itemgetter(1)))[0] == index:
            correct[0] += 1
        # Get top 5
        if index in dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[1]]).keys():
            correct[1] += 1
        if index in dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[2]]).keys():
            correct[2] += 1    
        total += 1
    print('\n******************************')
    print('***** English Caption retrieval *****')
    print('******************************')
    print('Total images assessed: ', total)
    print('******')
    print(f'Correct rank {ranks[0]}: ', correct[0])
    print(f'Correct rank {ranks[1]}: ', correct[1])
    print(f'Correct rank {ranks[2]}: ', correct[2])
    print('******')
    print(f"The total percentage rank {ranks[0]} is {correct[0]/total*100:.2f}%")
    print(f"The total percentage rank {ranks[1]} is {correct[1]/total*100:.2f}%")
    print(f"The total percentage rank {ranks[2]} is {correct[2]/total*100:.2f}%")    