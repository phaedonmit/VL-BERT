"""
Author: Faidon Mitzalis
Generates json file for training set with all 5 captions of flickr30k dataset
"""

captions_en = []
captions_de = []
urls = []
train_image_ids = []
frcnns = []
images = []

with open('/experiments/faidon/VL-BERT/data/flickr30k/utils/train.txt') as fp:
    for cnt, line in enumerate(fp):
        image_id = (line.split('.')[0])
        train_image_ids.append(image_id)


with open('/experiments/faidon/VL-BERT/data/flickr30k/5_captions/results_20130124.token') as fp:
    for cnt, line in enumerate(fp):
        line_array = line.split('\t')
        image_id = (line_array[0].split('#')[0].split('.')[0])
        if image_id in train_image_ids:
            images.append('train_image.zip@/'+ image_id +'.jpg')
            while len(image_id)<8:
                image_id = '0'+image_id
            frcnns.append('train_frcnn.zip@/'+ image_id +'.json')
            captions_en.append(line_array[1].strip())


import json
with open('/experiments/faidon/VL-BERT/data/flickr30k/train_5captions.json', 'w') as outfile:
    for cnt, (caption_en, image) in enumerate(zip(captions_en, images)):
            d = {'image':image, 'caption_en':caption_en}
            json.dump(d, outfile)
            outfile.write('\n')
            
with open('/experiments/faidon/VL-BERT/data/flickr30k/train_frcnn_5captions.json', 'w') as outfile:
    for cnt, (caption_en, image, frcnn) in enumerate(zip(captions_en, images, frcnns)):
            d = {'image':image, 'caption_en':caption_en, 'frcnn':frcnn}
            json.dump(d, outfile)
            outfile.write('\n')
