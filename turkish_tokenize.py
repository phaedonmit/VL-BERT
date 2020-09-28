"""
Author: Faidon Mitzalis
Edited from VL-BERT impelementation
Generates json file for training set with caption/images/frcnn for turkish flickr8k dataset
"""
import sys
sys.path.append("/data/faidon/VL-BERT")
import json
from external.pytorch_pretrained_bert import BertTokenizer

captions_en = []
captions_de = []
tokenized = []
urls = []
images = []
frcnns = []

tokenizer = BertTokenizer.from_pretrained('./model/pretrained_model/bert-base-multilingual-cased', do_lower_case=False)
with open("turkish_tokenized_CASED.txt", "w") as text_file:
    with open('./data/flickr30k/turkish8k/tasviret8k_captions.json', 'r') as fp:
        # for cnt, line in enumerate(fp):
        #     print(cnt)
        data = json.load(fp)
        for num, item in enumerate(data['images']):
            if item['split']=='train':
                for sentence in item['sentences']:
                    text_file.write('**********************************************\n')
                    text_file.write(sentence['raw']+'\n')
                    text_file.write(' '.join(tokenizer.tokenize(sentence['raw']))+'\n')

exit()

with open('train.txt') as fp:
    for cnt, line in enumerate(fp):
        image_id = (line.split('.')[0])
        images.append('train_image.zip@/'+ image_id +'.jpg')
        while len(image_id)<8:
            image_id = '0'+image_id
        frcnns.append('train_frcnn.zip@/'+ image_id +'.json')

with open('../train.en') as fp:
    for cnt, line in enumerate(fp):
        captions_en.append(line.strip())

with open('../train.de') as fp:
    for cnt, line in enumerate(fp):
        captions_de.append(line.strip())      

# Check lengths are the same  
assert(len(captions_en)==len(captions_de))

with open('../train.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image) in enumerate(zip(captions_en, captions_de, images)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de}
            json.dump(d, outfile)
            outfile.write('\n')
            
with open('../train_frcnn.json', 'w') as outfile:
    for cnt, (caption_en, caption_de, image, frcnn) in enumerate(zip(captions_en, captions_de, images, frcnns)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de, 'frcnn':frcnn}
            json.dump(d, outfile)
            outfile.write('\n')
