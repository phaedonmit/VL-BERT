"""
Author: Faidon Mitzalis
Edited from VL-BERT impelementation
Generates json file for training set with caption/images/frcnn for multi30k dataset
"""
import os
import json
import random 
from tqdm import tqdm

captions_en = []
captions_de = []

# ****************
# Step 1: Read all files
filepath_en = '../train.en'
filepath_de = '../train.de'

with open(filepath_en) as fp: 
    for cnt, line in tqdm(enumerate(fp)):
        captions_en.append(line.strip())
with open(filepath_de) as fp: 
    for cnt, line in enumerate(fp):
        captions_de.append(line.strip())


# Check lengths are the same  
assert(len(captions_en)==len(captions_de))


dataset = 'train'
with open('../' + dataset + '.json', 'w') as outfile:
    for caption_en, caption_de in zip(captions_en, captions_de):
            d = {'caption_en':caption_en, 'caption_de':caption_de}
            json.dump(d, outfile)
            outfile.write('\n')
            
