######
# Author: Faidon Mitzalis
# Date: June 2020
# Comments: Use results of testing to evaluate performance of image retrieval task
######

import json
import torch
import operator 

filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MLT/001_MTL_ende_with_vision/001_MTL_ende_with_vision_MLT_test2015.json"
# filepath = "/experiments/faidon/VL-BERT/checkpoints/itm_model19_LR1e_6_all.json"

with open(filepath) as json_file:
    data = json.load(json_file)
    correct = 0
    total=0
    captions_dict = {}
    image_dict = {}

    #******************************************************
    # Step 1: Read json to dictionaries
    
    # json file to nested dictionary (each caption with all images)
    for p in data:
        print(p)
        total += 1
        if p['word_de_id'] == p['logit']:
            correct += 1

    #******************************************************
    # Step 2: Get ranks image retrieval
    print('******************************')
    print('***** MLT Task    *****')
    print('******************************')
    print('Total words assessed: ', total)
    print(f'Correct translations retrieved: ', correct)
    print('******')
    print(f"Accuracy is {correct/total*100:.2f}%")
