"""
Author: Faidon Mitzalis
Edited from VL-BERT impelementation
Generates json file for training set with caption/images/frcnn for turkish flickr8k dataset
"""
import sys
sys.path.append("/data/faidon/VL-BERT")
import json
from external.pytorch_pretrained_bert import BertTokenizer
import jsonlines

captions_en = []
captions_de = []
tokenized = []
urls = []
images = []
frcnns = []

tokenizer = BertTokenizer.from_pretrained('./model/pretrained_model/bert-base-multilingual-cased', do_lower_case=False)
with open("tokenized_files/english_iwslt_ende_train_tokenized.txt", "w") as text_file:
    path = '/data/ozan/datasets/mmbert/iwslt14_en_de/train_frcnn.json'
        # for cnt, line in enumerate(fp):
        #     print(cnt)
    data = list(jsonlines.open(path))
    for num, item in enumerate(data):
                    #text_file.write('**********************************************\n')
                    #text_file.write(sentence['raw']+'\n')
        text_file.write(' '.join(tokenizer.tokenize(item['caption_en']))+'\n')

