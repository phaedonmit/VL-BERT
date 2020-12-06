# def check(*args):
#     [one, two, three] = args
#     print(len(args))
#     print(one)
# import torch
import json
import jsonlines

if __name__ == "__main__":
    # check('this', 'is', 'the')
    ann_file = "/data/ozan/datasets/mmbert/iwslt14_en_de/test_frcnn.json"
    simple_database = list(jsonlines.open(ann_file))
    example = simple_database[1]
    words = example['caption_de'].split(' ')
    with open('/data/faidon/VL-BERT/data/testing/test_frcnn.json', 'w') as outfile:
        for cnt,  word_en in enumerate((words)):
            d = {'caption_de': ' '.join(words[:cnt+1]),
                 'caption_en': example['caption_en']}
            json.dump(d, outfile)
            outfile.write('\n')
    print(example)
