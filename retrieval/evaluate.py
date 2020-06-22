import json
import torch
import operator 

filepath = "/experiments/faidon/VL-BERT/checkpoints/itm_evaluation_retrieval_test2015.json"

with open(filepath) as json_file:
    data = json.load(json_file)
    correct=0
    total=0
    captions_dict = {}
    for p in data:
        if p['caption_id'] in captions_dict.keys():
            captions_dict[p['caption_id']][p['image_ids']]= (torch.sigmoid(torch.tensor(p['logit'][0])).item())
        else:
            captions_dict[p['caption_id']] = {p['image_ids']: torch.sigmoid(torch.tensor(p['logit'][0])).item()}

    for index, value in captions_dict.items():
        print('-------')
        print('Top candiate is: ', (max(value.items(), key=operator.itemgetter(1))))
        print('Actual image id is: ', index)
        if (max(value.items(), key=operator.itemgetter(1)))[0] == index:
            correct += 1
        total += 1

    print('Total images: ', total)
    print('Correct images: ', correct)
    print(f"The total percentage is {correct/total*100}%")