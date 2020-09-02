######
# Author: Faidon Mitzalis
# Date: June 2020
# Comments: Run model with all caption-image pairs in the dataset
######

import os
import pprint
import shutil

import json
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from MLT.data.build import make_dataloader
from MLT.modules import *


@torch.no_grad()
def test_net2018(args, config, ckpt_path=None, save_path=None, save_name=None):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ckpt_path is None:
        _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
                                             split='train')
        model_prefix = os.path.join(train_output_path, config.MODEL_PREFIX)
        ckpt_path = '{}-best.model'.format(model_prefix)
        print('Use best checkpoint {}...'.format(ckpt_path))
    if save_path is None:
        logger, test_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TEST_IMAGE_SET,
                                                 split='test')
        save_path = test_output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy2(ckpt_path,
                 os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

    # ************
    # Step 1: Select model architecture and preload trained model
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # ************
    # Step 2: Create dataloader to include all caption-image pairs
    test_loader = make_dataloader(config, mode='test', distributed=False)
    test_dataset = test_loader.dataset
    test_database = test_dataset.database
    vocab = test_dataset.MLT_vocab

    # ************
    # Step 3: Run all pairs through model for inference
    word_de_ids = []
    words_de = []
    words_en = []
    captions_en = []
    captions_de = []
    logit_words = []
    logits = []
    model.eval()
    cur_id = 0
    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
        bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
        # for id in range(cur_id, min(cur_id + bs, len(test_database))):
        #     print(test_database[id])
        words_de.extend([test_database[id]['word_de'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        words_en.extend([test_database[id]['word_en'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        captions_en.extend([test_database[id]['caption_en'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        captions_de.extend([test_database[id]['caption_de'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        batch = to_cuda(batch)
        output = model(*batch)
        # FM note: output is tuple (outputs, loss)
        probs = F.softmax(output[0]['MLT_logits'].float(), dim=1)
        batch_size = probs.shape[0]
        logits.extend(probs.argmax(dim=1).detach().cpu().tolist())
        # word_de_ids.extend(output[0]['MLT_label'].detach().cpu().tolist())
        logit_words.extend([vocab[id] for id in logits[cur_id:min(cur_id + bs, len(test_database))]])

        cur_id += bs

        #     output = model(*batch)
        #     probs = F.softmax(output['label_logits'].float(), dim=1)
        #     batch_size = probs.shape[0]
        #     test_probs.append(probs.float().detach().cpu().numpy())
        #     test_ids.append([test_database[cur_id + k]['annot_id'] for k in range(batch_size)])
        # logits.extend(F.sigmoid(output[0]['relationship_logits']).detach().cpu().tolist())

    # ************
    # Step 3: Store all logit results in file for later evalution       
    result = [{'logit': l_id, 'word_en': word_en, 'word_de': word_de, 'word_pred': logit_word,
             'caption_en': caption_en, 'caption_de': caption_de} 
                for l_id, word_en, word_de, logit_word, caption_en, caption_de
                 in zip(logits, words_en, words_de, logit_words, captions_en, captions_de)]
    cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
    result_json_path = os.path.join(save_path, '{}_MLT_{}.json'.format(cfg_name if save_name is None else save_name,
                                                                        config.DATASET.TEST_IMAGE_SET))
    with open(result_json_path, 'w') as f:
        json.dump(result, f)
    print('result json saved to {}.'.format(result_json_path))
    return result_json_path
