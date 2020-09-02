######
# Author: Faidon Mitzalis
# Date: June 2020
# Comments: Use results of testing to evaluate performance of image retrieval task
######

import json
import torch
import operator 
import jsonlines

# filepath = "/experiments/faidon/VL-BERT/checkpoints/itm_evaluation_retrieval_test2015.json"
# filepath = "/experiments/faidon/VL-BERT/checkpoints/imt_model19_prec_5captions_LR1e6_all.json"
filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/itm_prec/015_prec_retrieval_mixed_5x_startTaskB/015_prec_retrieval_mixed_5x_startTaskB_German_retrieval_test2015.json"

test_images = list(jsonlines.open("/experiments/faidon/VL-BERT/data/flickr30k/test.json"))

with open(filepath) as json_file:
    data = json.load(json_file)
    ranks = [1, 5, 10]
    correct = [0, 0, 0]
    total=0
    captions_dict = {}
    image_dict = {}

    #******************************************************
    # Step 1: Read json to dictionaries
    
    # json file to nested dictionary (each caption with all images)
    for p in data:
        if p['caption_id'] in captions_dict.keys():
            captions_dict[p['caption_id']][p['image_ids']]= p['logit'][0]
        else:
            captions_dict[p['caption_id']] = {p['image_ids']: p['logit'][0]}
    # # json file to nested dictionary (each image with all caption)
        if p['image_ids'] in image_dict.keys():
            image_dict[p['image_ids']][p['caption_id']]= p['logit'][0]
        else:
            image_dict[p['image_ids']] = {p['caption_id']: p['logit'][0]}

    #******************************************************
    # Step 2: Get ranks image retrieval
    # get r1, r5 and r10 for each capion
    for index, value in captions_dict.items():
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
        # TODO: REMOVE: get examples
        print("caption: ", test_images[index]['caption_de'])
        print("caption: ", test_images[index]['caption_en'])
        print("sentence: ", index)
        print("images: ", dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[1]]).keys())
        for image in dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[1]]).keys():
            print('image line: ', image)
            print('image id: ', test_images[image]['image'])
        if total==480:
            exit()
        total += 1
    print('******************************')
    print('***** Image retrieval    *****')
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
    for index, value in image_dict.items():
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
        # # TODO: REMOVE: get examples
        # print("image: ", index)
        # print("sentences: ", dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[1]]).keys())
        # print("image id: ", test_images[index]['image'])
        # for image in dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[1]]).keys():
        #     print('image line: ', image)
        #     print('caption de: ', test_images[image]['caption_de'])
        #     print('caption en: ', test_images[image]['caption_en'])
        # if total==51:
        #     exit()
        total += 1
    print('\n******************************')
    print('***** Caption retrieval  *****')
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