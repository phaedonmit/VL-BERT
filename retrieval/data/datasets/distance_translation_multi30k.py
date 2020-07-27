import random
import os
import time
import json
import jsonlines
from PIL import Image
import base64
import numpy as np
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist


class Distance_Translation_Multi30kDataset(Dataset):
    def __init__(self, ann_file, image_set, root_path, data_path, seq_len=64,
                 with_precomputed_visual_feat=False, mask_raw_pixels=True,
                 with_rel_task=True, with_mlm_task=False, with_mvrc_task=False,
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False,
                 aspect_grouping=False, languages_used='first', **kwargs):
        """
        Conceptual Captions Dataset

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(Distance_Translation_Multi30kDataset, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'
        # TODO: need to remove this to allows testing
        # assert not test_mode

        annot = {'train': 'train_frcnn.json',
                 'val': 'val_frcnn.json',
                 'test2015': 'test_frcnn.json'}

        self.seq_len = seq_len
        self.with_rel_task = with_rel_task
        self.with_mlm_task = with_mlm_task
        self.with_mvrc_task = with_mvrc_task
        self.data_path = data_path
        self.root_path = root_path
        self.ann_file = os.path.join(data_path, annot[image_set])
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.mask_raw_pixels = mask_raw_pixels
        self.image_set = image_set
        self.transform = transform
        self.test_mode = test_mode
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        #FM edit: added option for how many captions
        self.languages_used = languages_used
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        self.zipreader = ZipReader()

        # FM: Customise for multi30k dataset - only used for inference
        self.database = list(jsonlines.open(self.ann_file))
        # if not self.test_mode:
        #     self.database = list(jsonlines.open(self.ann_file))
        # # FM edit: create dataset for test mode 
        # else:
        #     self.simple_database = list(jsonlines.open(self.ann_file))
        #     # create database cross-coupling each caption_en with all captions_de
        #     self.database = []
        #     db_index = 0
        #     for x, idb_x in enumerate(self.simple_database):
        #         for y, idb_y in enumerate(self.simple_database):                    
        #             self.database.append({})
        #             self.database[db_index]['label'] = 1.0 if x==y else 0.0
        #             self.database[db_index]['caption_en'] = self.simple_database[x]['caption_en']
        #             self.database[db_index]['caption_de'] = self.simple_database[y]['caption_de']
        #             self.database[db_index]['caption_en_index'] = x
        #             self.database[db_index]['caption_de_index'] = y
        #             db_index += 1

        if self.aspect_grouping:
            assert False, "not support aspect grouping currently!"
            self.group_ids = self.group_aspect(self.database)

        print('mask_raw_pixels: ', self.mask_raw_pixels)

    @property
    def data_names(self):
        return ['text', 'relationship_label', 'mlm_labels']

    def __getitem__(self, index):
        idb = self.database[index]

        # # indeces for inference
        # caption_en_index = idb['caption_en_index'] if self.test_mode else 0
        # caption_de_index = idb['caption_de_index'] if self.test_mode else 0

        # Task #1: Caption-Image Relationship Prediction
        _p = random.random()
        if not self.test_mode:
            if _p < 0.5:
                relationship_label = 1.0
                caption_en = idb['caption_en']
                caption_de = idb['caption_de']
            else:
                relationship_label = 0.0
                rand_index = random.randrange(0, len(self.database))
                while rand_index == index:
                    rand_index = random.randrange(0, len(self.database))
                # caption_en and image match, german caption is random
                caption_en = idb['caption_en']
                caption_de = self.database[rand_index]['caption_de']
        # for inference
        else:
            relationship_label = 1
            caption_en = idb['caption_en']
            caption_de = idb['caption_de']

         # FM edit: add captions
        caption_tokens_en = self.tokenizer.tokenize(caption_en)
        caption_tokens_de = self.tokenizer.tokenize(caption_de)
        mlm_labels_en = [-1] * len(caption_tokens_en)
        mlm_labels_de = [-1] * len(caption_tokens_de)
        
        # FM edit: captions of both languages exist in all cases
        if self.languages_used == 'first':
            text_tokens = ['[CLS]'] + caption_tokens_en + ['[SEP]']
            mlm_labels = [-1] + mlm_labels_en + [-1]
        elif self.languages_used == 'second':
            text_tokens = ['[CLS]'] + caption_tokens_de + ['[SEP]']
            mlm_labels = [-1] + mlm_labels_de + [-1]
        else:
            text_tokens = ['[CLS]'] + caption_tokens_en + ['[SEP]'] + caption_tokens_de + ['[SEP]']
            mlm_labels = [-1] + mlm_labels_en + [-1] + mlm_labels_de + [-1]



        # convert tokens to ids 
        text = self.tokenizer.convert_tokens_to_ids(text_tokens)

        # truncate seq to max len
        if len(text)  > self.seq_len:
            text_len_keep = len(text)
            while (text_len_keep) > self.seq_len and (text_len_keep > 0):
                text_len_keep -= 1
            if text_len_keep < 2:
                text_len_keep = 2
            text = text[:(text_len_keep - 1)] + [text[-1]]

        return text, relationship_label, mlm_labels

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)

