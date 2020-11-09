"""
Author: Faidon Mitzalis
Edited from VL-BERT impelementation
Generates json file for training set with caption/images/frcnn for multi30k dataset
"""
import sys

captions_en = []
captions_de = []
urls = []
images = []
frcnns = []

lng1 = str(sys.argv[1])
lng2 = str(sys.argv[2])
dataset = str(sys.argv[3])
print('Language 1: ', lng1)
print('Language 2: ', lng2)
print('Dataset: ', dataset)

with open(dataset+'.txt') as fp:
    for cnt, line in enumerate(fp):
        image_id = (line.split('.')[0])
        images.append(dataset+'_image.zip@/'+ image_id +'.jpg')
        while len(image_id)<8:
            image_id = '0'+image_id
        frcnns.append(dataset+'_frcnn.zip@/'+ image_id +'.json')

with open('../'+ dataset +'.'+lng1) as fp:
    for cnt, line in enumerate(fp):
        captions_en.append(line.strip())

with open('../'+ dataset +'.'+lng2) as fp:
    for cnt, line in enumerate(fp):
        captions_de.append(line.strip())      

# Check lengths are the same  
assert(len(captions_en)==len(captions_de))

import json
with open('../'+dataset+'_'+lng2+'.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image) in enumerate(zip(captions_en, captions_de, images)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de}
            json.dump(d, outfile)
            outfile.write('\n')
            
with open('../'+dataset+'_'+lng2+'_frcnn.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image, frcnn) in enumerate(zip(captions_en, captions_de, images, frcnns)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de, 'frcnn':frcnn}
            json.dump(d, outfile)
            outfile.write('\n')
