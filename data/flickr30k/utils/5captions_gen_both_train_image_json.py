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

# load images (images repeat 5 times):
with open('/experiments/faidon/VL-BERT/data/flickr30k/utils/train.txt') as fp:
    for cnt, line in enumerate(fp):
        image_id = (line.split('.')[0])
        images.append('train_image.zip@/'+ image_id +'.jpg')
        while len(image_id)<8:
            image_id = '0'+image_id
        frcnns.append('train_frcnn.zip@/'+ image_id +'.json')
images = images*5
frcnns = frcnns*5


# add all 5 english german caption sets
for lang in ['en', 'de']:
    for i in range(1,6):
        # load captions:
        filepath = f'/experiments/faidon/multi30k-dataset/data/task2/tok/train.lc.norm.tok.{i}.{lang}'
        with open(filepath) as fp:
            for cnt, line in enumerate(fp):                
                if lang=='en':
                    captions_en.append(line.strip())
                else:
                    captions_de.append(line.strip())

# Check lengths are equal:
assert(len(captions_en)==len(captions_de)==len(images)==len(frcnns))


import json
with open('/experiments/faidon/VL-BERT/data/flickr30k/train_5captions_both.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image) in enumerate(zip(captions_en, captions_de, images)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de}
            json.dump(d, outfile)
            outfile.write('\n')
            
with open('/experiments/faidon/VL-BERT/data/flickr30k/train_frcnn_5captions_both.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image, frcnn) in enumerate(zip(captions_en, captions_de, images, frcnns)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de, 'frcnn':frcnn}
            json.dump(d, outfile)
            outfile.write('\n')
