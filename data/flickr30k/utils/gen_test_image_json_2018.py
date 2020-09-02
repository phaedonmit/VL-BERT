"""
Author: Faidon Mitzalis
Edited from VL-BERT impelementation
Generates json file for validation set with caption/images/frcnn for multi30k dataset
"""

captions_en = []
captions_de = []
urls = []
images = []
frcnns = []

with open('test_2018_flickr.txt') as fp:
    for cnt, line in enumerate(fp):
        image_id = (line.split('.')[0])
        image_id = image_id.split('_')[0]
        images.append('test_image2018_renamed.zip@/'+ image_id +'.jpg')
        while len(image_id)<8:
            image_id = '0'+image_id        
        frcnns.append('test2018_renamed_frcnn.zip@/'+ image_id +'.json')

with open('../test_2018.en') as fp:
    for cnt, line in enumerate(fp):
        captions_en.append(line.strip())

with open('../test_2018.en') as fp:
    for cnt, line in enumerate(fp):
        captions_de.append(line.strip())      

# Check length is the same  
assert(len(captions_en)==len(captions_de))

import json
with open('../test2018.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image) in enumerate(zip(captions_en, captions_de, images)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de}
            json.dump(d, outfile)
            outfile.write('\n')
            
with open('../test_frcnn2018.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image, frcnn) in enumerate(zip(captions_en, captions_de, images, frcnns)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de, 'frcnn':frcnn}
            json.dump(d, outfile)
            outfile.write('\n')
