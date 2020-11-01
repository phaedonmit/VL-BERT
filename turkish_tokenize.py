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

