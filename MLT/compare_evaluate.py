######
# Author: Faidon Mitzalis
# Date: June 2020
# Comments: Use results of testing from two experiments to compare incorrect results
######

import json
import torch
import operator 

filepath_novision = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MLT/002_MLT_ende_no_vision/002_MLT_ende_no_vision_MLT_test2015.json"
filepath_vision = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MLT/001_MLT_ende_with_vision/001_MTL_ende_with_vision_MLT_test2015.json"

with open(filepath_novision) as json_file:
    data = json.load(json_file)
    correct = 0
    total=0
    words_en = []
    words_de = []
    captions_en = []
    captions_de = []
    predicted = []
    nid_found = []
    #******************************************************
    # Step 1: Read json to lists
    
    # json file to nested dictionary (each caption with all images)
    for nid, p in enumerate(data):
        # print(p)
        total += 1
        if p['word_de_id'] == p['logit']:
            nid_found.append(nid)
        captions_en.append(p['caption_en'])
        captions_de.append(p['caption_de'])    
        words_en.append(p['word_en'])
        words_de.append(p['word_de'])
        predicted.append(p['word_pred'])


with open(filepath_vision) as json_file:
    data = json.load(json_file)
    #******************************************************
    # Step 1: Read json to lists
    
    # json file to nested dictionary (each caption with all images)
    for nid, p in enumerate(data):
        # print(p)
        total += 1
        if p['word_de_id'] == p['logit'] and (nid not in nid_found):
            correct += 1
            print('***********************')
            print(p['caption_en'])
            print(p['caption_de'])
            print('word en: ', p['word_en'])
            print('word de: ', p['word_de'])
            print('predicted: ', p['word_pred'])
            print('--- No Vision')
            print('word predicted: ', predicted[nid])


print('Total improved: ', correct)