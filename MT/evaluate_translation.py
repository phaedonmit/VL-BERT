######
# Author: Faidon Mitzalis
# Date: June 2020
# Comments: Use results of testing to evaluate performance of MT translation task
######

import json
import torch
import operator 
import sacrebleu
import unidecode

model = "GLOBAL002_IC_test2016_epoch007"
filepath = "/experiments/faidon/test/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_MT_LR6_global/GLOBAL_001_IC_start_taskA_epoch007_MT_test2015.json"

with open(filepath) as json_file:
    data = json.load(json_file)
    correct = 0
    total=0
    captions_dict = {}
    image_dict = {}
    references = []
    hypotheses = []

    #******************************************************
    # Step 1: Read json to dictionaries
    
    # json file to nested dictionary (each caption with all images)
    for p in data:
        # print(p)
        # total += 1
        # if p['word_de_id'] == p['logit']:
        #     correct += 1
        # else:
        print('***********************')
        print('source: ', p['caption_en'])
        # remove accents
        reference = unidecode.unidecode(p['caption_de'])
        # reference = (p['caption_de'])
        # convert ß to ss (equivalent in german)
        hypothesis = p['generated_sentence'].replace('ß','ss')
        print('reference: ', reference)
        print('generated: ', hypothesis)
        references.append(reference)
        hypotheses.append(hypothesis)

    

    #******************************************************
    # # Step 2: Get BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(bleu.score)

    with open(model+'_ref.txt', 'w') as f:
        for ref in references:
            f.write("%s\n" % ref)
    with open(model+'_hyp.txt', 'w') as f:
        for hyp in hypotheses:
            f.write("%s\n" % hyp)        