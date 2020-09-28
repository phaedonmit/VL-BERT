"""
Author: Faidon Mitzalis
Edited from VL-BERT impelementation
Generates json file for training set with caption/images/frcnn for turkish flickr8k dataset
"""

import json

captions_en = []
captions_de = []
tokenized = []
urls = []
images = []
frcnns = []


with open('../turkish8k/tasviret8k_captions.json', 'r') as fp:
    data = json.load(fp)
    for num, item in enumerate(data['images']):
        if item['split']=='test':
            image_id = item['filename'].split('_')[0]
            for sentence in item['sentences']:
                images.append('test_image.zip@/'+ image_id +'.jpg')
                while len(image_id)<8:
                    image_id = '0'+image_id
                frcnns.append('test_frcnn.zip@/'+ image_id +'.json')
                captions_de.append(sentence['raw'])
                # print(images[-1])
                # print(captions_de[-1])

with open('../test_turkish.json', 'w') as outfile:
    for cnt, (caption_de, image) in enumerate(zip(captions_de, images)):
            d = {'image':image, 'caption_de':caption_de}
            json.dump(d, outfile)
            outfile.write('\n')
            
with open('../test_turkish_frcnn.json', 'w') as outfile:
    for cnt, (caption_de, image, frcnn) in enumerate(zip(captions_de, images, frcnns)):
            d = {'image':image, 'caption_de':caption_de, 'frcnn':frcnn}
            json.dump(d, outfile)
            outfile.write('\n')
