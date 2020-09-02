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


class Multi30kDataset(Dataset):
    def __init__(self, ann_file, image_set, root_path, data_path, seq_len=64,
                 with_precomputed_visual_feat=False, mask_raw_pixels=True,
                 with_rel_task=True, with_mlm_task=False, with_mvrc_task=False,
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False,
                 aspect_grouping=False, languages_used='first', MLT_vocab='bert-base-german-cased-vocab.txt', **kwargs):
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
        super(Multi30kDataset, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'
        # TODO: need to remove this to allows testing
        # assert not test_mode

        annot = {'train': 'train_MLT_frcnn.json',
                 'val': 'val_MLT_frcnn.json',
                 'test2015': 'test_MLT_frcnn.json',
                 'test2018': 'test_MLT_2018_renamed_frcnn.json'}

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

        # FM: Customise for multi30k dataset
        self.database = list(jsonlines.open(self.ann_file))
        if not self.zip_mode:
            for i, idb in enumerate(self.database):
                self.database[i]['frcnn'] = idb['frcnn'].replace('.zip@', '')\
                    .replace('.0', '').replace('.1', '').replace('.2', '').replace('.3', '')
                self.database[i]['image'] = idb['image'].replace('.zip@', '')


        if self.aspect_grouping:
            assert False, "not support aspect grouping currently!"
            self.group_ids = self.group_aspect(self.database)

        print('mask_raw_pixels: ', self.mask_raw_pixels)

        #FM: initialise vocabulary for output
        self.MLT_vocab_path = os.path.join(root_path, 'model/pretrained_model', MLT_vocab)
        self.MLT_vocab = []
        with open(self.MLT_vocab_path) as fp:
            for cnt, line in enumerate(fp):
                self.MLT_vocab.append(line.strip())


    @property
    def data_names(self):
        return ['image', 'boxes', 'im_info', 'text',
                'relationship_label', 'mlm_labels', 'mvrc_ops', 'mvrc_labels', 'word_de_id']

    def __getitem__(self, index):
        idb = self.database[index]

        # image data
        # IN ALL CASES: boxes and cls scores are available for each image
        frcnn_data = self._load_json(os.path.join(self.data_path, idb['frcnn']))
        boxes = np.frombuffer(self.b64_decode(frcnn_data['boxes']),
                              dtype=np.float32).reshape((frcnn_data['num_boxes'], -1))
        boxes_cls_scores = np.frombuffer(self.b64_decode(frcnn_data['classes']),
                                         dtype=np.float32).reshape((frcnn_data['num_boxes'], -1))
        boxes_max_conf = boxes_cls_scores.max(axis=1)
        inds = np.argsort(boxes_max_conf)[::-1]
        boxes = boxes[inds]
        boxes_cls_scores = boxes_cls_scores[inds]
        boxes = torch.as_tensor(boxes)

        # load precomputed features or the whole image depending on setup
        if self.with_precomputed_visual_feat:
            image = None
            w0, h0 = frcnn_data['image_w'], frcnn_data['image_h']
            boxes_features = np.frombuffer(self.b64_decode(frcnn_data['features']),
                                           dtype=np.float32).reshape((frcnn_data['num_boxes'], -1))
            boxes_features = boxes_features[inds]
            boxes_features = torch.as_tensor(boxes_features)
        else:
            try:
                image = self._load_image(os.path.join(self.data_path, idb['image']))
                w0, h0 = image.size
            except:
                print("Failed to load image {}, use zero image!".format(idb['image']))
                image = None
                w0, h0 = frcnn_data['image_w'], frcnn_data['image_h']

        # append whole image to tensor of boxes (used for all linguistic tokens)
        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1.0, h0 - 1.0]])
            boxes = torch.cat((image_box, boxes), dim=0)
            if self.with_precomputed_visual_feat:
                image_box_feat = boxes_features.mean(dim=0, keepdim=True)
                boxes_features = torch.cat((image_box_feat, boxes_features), dim=0)

        # transform
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        if self.transform is not None:
            image, boxes, _, im_info = self.transform(image, boxes, None, im_info)

        if image is None and (not self.with_precomputed_visual_feat):
            w = int(im_info[0].item())
            h = int(im_info[1].item())
            image = im_info.new_zeros((3, h, w), dtype=torch.float)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w-1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h-1)

        # FM edit: remove - Task #1: Caption-Image Relationship Prediction
        word_en = idb['word_en']
        word_de = idb['word_de']
        caption_en = idb['caption_en']
        caption_de = idb['caption_de']
        
         # FM edit: add captions - tokenise words
        caption_tokens_en = self.tokenizer.tokenize(caption_en)
        # caption_tokens_de = self.tokenizer.tokenize(caption_de)
        word_tokens_en = self.tokenizer.tokenize(word_en)
        # word_tokens_de = self.tokenizer.tokenize(word_de)
        mlm_labels_en = [-1] * len(caption_tokens_en)
        mlm_labels_word_en = [-1] * len(caption_tokens_en)
        # mlm_labels_word_de = [-1] * len(caption_tokens_de)
        # mlm_labels_de = [-1] * len(caption_tokens_de)
        
        text_tokens = ['[CLS]'] + word_tokens_en + ['[SEP]'] + caption_tokens_en + ['[SEP]']
        mlm_labels = [-1] + mlm_labels_word_en + [-1] + mlm_labels_en + [-1]

        # relationship label - not used
        relationship_label = 1

        # Construct boxes
        mvrc_ops = [0] * boxes.shape[0]
        mvrc_labels = [np.zeros_like(boxes_cls_scores[0])] * boxes.shape[0]

        # store labels for masked regions
        mvrc_labels = np.stack(mvrc_labels, axis=0)

        # convert tokens to ids 
        text = self.tokenizer.convert_tokens_to_ids(text_tokens)

        if self.with_precomputed_visual_feat:
            boxes = torch.cat((boxes, boxes_features), dim=1)

        # truncate seq to max len
        if len(text) + len(boxes) > self.seq_len:
            text_len_keep = len(text)
            box_len_keep = len(boxes)
            while (text_len_keep + box_len_keep) > self.seq_len and (text_len_keep > 0) and (box_len_keep > 0):
                if box_len_keep > text_len_keep:
                    box_len_keep -= 1
                else:
                    text_len_keep -= 1
            if text_len_keep < 2:
                text_len_keep = 2
            if box_len_keep < 1:
                box_len_keep = 1
            boxes = boxes[:box_len_keep]
            text = text[:(text_len_keep - 1)] + [text[-1]]
            mlm_labels = mlm_labels[:(text_len_keep - 1)] + [mlm_labels[-1]]
            mvrc_ops = mvrc_ops[:box_len_keep]
            mvrc_labels = mvrc_labels[:box_len_keep]

        # FM edit: convert word_de to class
        word_de_id = self.MLT_vocab.index(word_de)



        return image, boxes, im_info, text, relationship_label, mlm_labels, mvrc_ops, mvrc_labels, word_de_id

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

