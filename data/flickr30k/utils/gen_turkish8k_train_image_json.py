"""
Author: Faidon Mitzalis
Edited from VL-BERT impelementation
Generates json file for training set with caption/images/frcnn for turkish flickr8k dataset
"""

import json
import os 

captions_en = []
captions_de = []
tokenized = []
urls = []
images = []
frcnns = []

json_files = [pos_json for pos_json in os.listdir('/data/faidon/VL-BERT/data/flickr30k/train_frcnn') if pos_json.endswith('.json')]

found = 0 
not_found = 0
with open('../turkish8k/tasviret8k_captions.json', 'r') as fp:
    data = json.load(fp)
    for num, item in enumerate(data['images']):
        if item['split']=='train':
            image_id = item['filename'].split('_')[0]
            if image_id+'.json' in json_files:
                found += 1
                for sentence in item['sentences']:
                    images.append('train_image.zip@/'+ image_id +'.jpg')
                    while len(image_id)<8:
                        image_id = '0'+image_id
                    frcnns.append('train_frcnn.zip@/'+ image_id +'.json')
                    captions_de.append(sentence['raw'])
                    # print(images[-1])
                    # print(captions_de[-1])
            else:
                not_found+=1


with open('../train_turkish.json', 'w') as outfile:
    for cnt, (caption_de, image) in enumerate(zip(captions_de, images)):
            d = {'image':image, 'caption_de':caption_de}
            json.dump(d, outfile)
            outfile.write('\n')
            
with open('../train_turkish_frcnn.json', 'w') as outfile:
    for cnt, (caption_de, image, frcnn) in enumerate(zip(captions_de, images, frcnns)):
            d = {'image':image, 'caption_de':caption_de, 'frcnn':frcnn}
            json.dump(d, outfile)
            outfile.write('\n')