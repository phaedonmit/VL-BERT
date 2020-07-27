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


class CC_and_Flickr30kDataset(Dataset):
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
        super(CC_and_Flickr30kDataset, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'
        # TODO: need to remove this to allows testing
        # assert not test_mode

        annot = {'train': ['train_frcnn_5captions.json', 'train_frcnn.json'],
                 'val': 'val_frcnn.json',
                 'test2015': 'test_frcnn.json'}

        self.seq_len = seq_len
        self.with_rel_task = with_rel_task
        self.with_mlm_task = with_mlm_task
        self.with_mvrc_task = with_mvrc_task
        self.data_path = data_path
        self.root_path = root_path
        if image_set == 'train':
            self.ann_file = [os.path.join(data_path, annot[image_set][0]), 
                         os.path.join(os.path.dirname(data_path), 'conceptual-captions', annot[image_set][1])]
        else:
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
        if not self.test_mode:
            # load flickr30k
            if self.image_set=='train':
                self.database_first = list(jsonlines.open(self.ann_file[0]))
                if not self.zip_mode:
                    for i, idb in enumerate(self.database_first):
                        self.database_first[i]['frcnn'] = idb['frcnn'].replace('.zip@', '')\
                            .replace('.0', '').replace('.1', '').replace('.2', '').replace('.3', '')
                        self.database_first[i]['image'] = idb['image'].replace('.zip@', '')
                        self.database_first[i]['dataset'] = 'flickr30k'
                # load conceptual captions
                self.database_second = list(jsonlines.open(self.ann_file[1]))
                if not self.zip_mode:
                    for i, idb in enumerate(self.database_second):
                        self.database_second[i]['frcnn'] = idb['frcnn'].replace('.zip@', '')\
                            .replace('.0', '').replace('.1', '').replace('.2', '').replace('.3', '')
                        self.database_second[i]['image'] = idb['image'].replace('.zip@', '')
                        self.database_second[i]['dataset'] = 'cc'

                # concatenate two datasets    
                self.database = self.database_first + self.database_second
            else:
                self.database = list(jsonlines.open(self.ann_file))
                if not self.zip_mode:
                    for i, idb in enumerate(self.database):
                        self.database[i]['frcnn'] = idb['frcnn'].replace('.zip@', '')\
                            .replace('.0', '').replace('.1', '').replace('.2', '').replace('.3', '')
                        self.database[i]['image'] = idb['image'].replace('.zip@', '')
                        self.database[i]['dataset'] = 'flickr30k'                
        # FM edit: create dataset for test mode 
        else:
            self.simple_database = list(jsonlines.open(self.ann_file))
            if not self.zip_mode:
                for i, idb in enumerate(self.simple_database):
                    self.simple_database[i]['frcnn'] = idb['frcnn'].replace('.zip@', '')\
                        .replace('.0', '').replace('.1', '').replace('.2', '').replace('.3', '')
                    self.simple_database[i]['image'] = idb['image'].replace('.zip@', '')
            # create database cross-coupling each caption with all images
            self.database = []
            db_index = 0
            for x, idb_x in enumerate(self.simple_database):
                for y, idb_y in enumerate(self.simple_database):                    
                    self.database.append({})
                    self.database[db_index]['label'] = 1.0 if x==y else 0.0
                    self.database[db_index]['caption_en'] = self.simple_database[x]['caption_en']
                    self.database[db_index]['image'] = self.simple_database[y]['image']
                    self.database[db_index]['frcnn'] = self.simple_database[y]['frcnn']
                    self.database[db_index]['caption_index'] = x
                    self.database[db_index]['image_index'] = y
                    self.database[db_index]['dataset'] = 'flickr30k'
                    db_index += 1

        if self.aspect_grouping:
            assert False, "not support aspect grouping currently!"
            self.group_ids = self.group_aspect(self.database)

        print('mask_raw_pixels: ', self.mask_raw_pixels)

    @property
    def data_names(self):
        return ['image', 'boxes', 'im_info', 'text',
                'relationship_label', 'mlm_labels', 'mvrc_ops', 'mvrc_labels', 'caption_index', 'image_index']

    def __getitem__(self, index):
        idb = self.database[index]

        # indeces for inference
        image_index = idb['image_index'] if self.test_mode else 0
        caption_index = idb['caption_index'] if self.test_mode else 0

        # image data
        # IN ALL CASES: boxes and cls scores are available for each image
        if idb['dataset'] == 'flickr30k':
            frcnn_data = self._load_json(os.path.join(self.data_path, idb['frcnn']))
        else:
            frcnn_data = self._load_json(os.path.join(os.path.dirname(self.data_path), 'conceptual-captions', idb['frcnn']))


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
                if idb['dataset']=='flickr30k':
                    image = self._load_image(os.path.join(self.data_path, idb['image']))
                else:
                    image = self._load_image(os.path.join(os.path.dirname(self.data_path), 'conceptual-captions', idb['image']))
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

        # Task #1: Caption-Image Relationship Prediction
        _p = random.random()
        if not self.test_mode:
            if _p < 0.5:
                relationship_label = 1.0
                if idb['dataset']=='flickr30k':
                    caption_en = idb['caption_en']
                else:
                    caption_en = ' '.join(idb['caption'])
            else:
                relationship_label = 0.0
                rand_index = random.randrange(0, len(self.database))
                while rand_index == index:
                    rand_index = random.randrange(0, len(self.database))
                if self.database[rand_index]['dataset']=='flickr30k':
                    caption_en = self.database[rand_index]['caption_en']
                else:
                    caption_en = ' '.join(self.database[rand_index]['caption'])
        # for inference
        else:
            relationship_label = idb['label']
            caption_en = idb['caption_en']

        # FM edit: add captions
        caption_tokens_en = self.tokenizer.tokenize(caption_en)
        mlm_labels_en = [-1] * len(caption_tokens_en)
        
        if self.languages_used == 'first':
            text_tokens = ['[CLS]'] + caption_tokens_en + ['[SEP]']
            mlm_labels = [-1] + mlm_labels_en + [-1]

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

        return image, boxes, im_info, text, relationship_label, mlm_labels, mvrc_ops, mvrc_labels, caption_index, image_index

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

