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

# filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MT/007_MT_LR6_with_vision_no_pretraining/007_MT_LR6_with_vision_no_pretraining_MT_test2015.json"
# 002 model
# filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MT/002_MT_LR6_with_vision_last_token/002_MT_LR6_with_vision_last_token_NOSTOP_MT_test2015.json"
# model = "006_model"
# filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MT/006_MT_LR6_with_vision_startTaskC/006_MT_LR6_with_vision_startTaskC_MT_test2015.json"
# model = "005_model"
# filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MT/005_MT_LR6_with_vision_startTaskB/005_MT_LR6_with_vision_startTaskB_MT_test2015.json"
# model = "007_model"
# filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MT/007_MT_LR6_with_vision_no_pretraining/007_MT_LR6_with_vision_no_pretraining_MT_test2015.json"
# model = "004_model"
# filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MT/004_MT_LR6_with_vision_encdec/004_MT_LR6_enc_dec_019_MT_test2015.json"
# model = "003_model"
# filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MT/003_MT_LR6_no_vision_last_token/003_MT_LR6_no_vision_last_token_MT_test2015.json"
model = "002_model"
filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MT/002_MT_LR6_with_vision_last_token/002_MT_LR6_with_vision_last_token_NOSTOP_MT_test2015.json"

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