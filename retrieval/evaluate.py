import json
import torch
import operator 

filepath = "/experiments/faidon/VL-BERT/checkpoints/itm_evaluation_retrieval_test2015.json"

with open(filepath) as json_file:
    data = json.load(json_file)
    ranks = [1, 5, 10]
    correct = [0, 0, 0]
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
            correct[0] += 1
        # Get top 5
        if index in dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[1]]).keys():
            correct[1] += 1
        if index in dict(sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:ranks[2]]).keys():
            correct[2] += 1
       
        total += 1

    print('Total images: ', total)
    print(f'Correct rank {ranks[0]}: ', correct[0])
    print(f'Correct rank {ranks[1]}: ', correct[1])
    print(f'Correct rank {ranks[2]}: ', correct[2])
    print(f"The total percentage rank {ranks[0]} is {correct[0]/total*100}%")
    print(f"The total percentage rank {ranks[1]} is {correct[1]/total*100}%")
    print(f"The total percentage rank {ranks[2]} is {correct[2]/total*100}%")