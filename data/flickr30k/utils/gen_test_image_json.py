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

with open('test_2016_flickr.txt') as fp:
    for cnt, line in enumerate(fp):
        image_id = (line.split('.')[0])
        images.append('test_image.zip@/'+ image_id +'.jpg')
        while len(image_id)<8:
            image_id = '0'+image_id        
        frcnns.append('test_frcnn.zip@/'+ image_id +'.json')

with open('../test_2016.en') as fp:
    for cnt, line in enumerate(fp):
        captions_en.append(line.strip())

with open('../test_2016.de') as fp:
    for cnt, line in enumerate(fp):
        captions_de.append(line.strip())      

# Check length is the same  
assert(len(captions_en)==len(captions_de))

import json
with open('../test.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image) in enumerate(zip(captions_en, captions_de, images)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de}
            json.dump(d, outfile)
            outfile.write('\n')
            
with open('../test_frcnn.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image, frcnn) in enumerate(zip(captions_en, captions_de, images, frcnns)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de, 'frcnn':frcnn}
            json.dump(d, outfile)
            outfile.write('\n')
