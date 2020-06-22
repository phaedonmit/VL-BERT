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
from retrieval.data.build import make_dataloader
from retrieval.modules import *


@torch.no_grad()
def test_net(args, config, ckpt_path=None, save_path=None, save_name=None):
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

    # get network
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # TODO: edit dataloader, need to set mode to test, or change data file
    # loader
    # test_loader = make_dataloader(config, mode='test', distributed=False)
    test_loader = make_dataloader(config, mode='test', distributed=False)
    test_dataset = test_loader.dataset
    test_database = test_dataset.database

    # TODO: tailor for captions
    #       First run once with each image
    #       Second run with cross testing each caption with all images
    # test
    caption_ids = []
    image_ids = []
    logits = []
    model.eval()
    cur_id = 0
    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
    # for nbatch, batch in tqdm(enumerate(test_loader)):
        bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
        # caption_ids.extend([test_database[id]['question_id'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        caption_ids.extend([test_database[id]['caption_index'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        image_ids.extend([test_database[id]['image_index'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        batch = to_cuda(batch)
        output = model(*batch)
        # answer_ids.extend(output['label_logits'].argmax(dim=1).detach().cpu().tolist())
        logits.extend(output[0]['relationship_logits'].detach().cpu().tolist())
        cur_id += bs
        if nbatch>300:
            break

    result = [{'caption_id': c_id, 'image_ids': i_id, 'logit': l_id} for c_id, i_id, l_id in zip(caption_ids, image_ids, logits)]
    # result = [{'caption_id': c_id, 'image_ids': test_dataset.answer_vocab[i_id]} for c_id, i_id in zip(caption_ids, image_ids)]

    cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
    result_json_path = os.path.join(save_path, '{}_retrieval_{}.json'.format(cfg_name if save_name is None else save_name,
                                                                        config.DATASET.TEST_IMAGE_SET))
    with open(result_json_path, 'w') as f:
        json.dump(result, f)
    print('result json saved to {}.'.format(result_json_path))
    return result_json_path