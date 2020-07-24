'''
Author: Faidon Mitzalis
Adapted from VL-BERT
Comments: Simple adaptation to derive output vector for a sentence
'''


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBertForDistance
from common.utils.misc import soft_cross_entropy

BERT_WEIGHTS_NAME = 'pytorch_model.bin'



class ResNetVLBERTDistanceTranslationNoVision(Module):
    def __init__(self, config):

        super(ResNetVLBERTDistanceTranslationNoVision, self).__init__(config)

        # Constructs/initialises model elements
        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        if config.NETWORK.IMAGE_FEAT_PRECOMPUTED or (not config.NETWORK.MASK_RAW_PIXELS):
            self.object_mask_visual_embedding = nn.Embedding(1, 2048)
        if config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.aux_text_visual_embedding = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        
        # Can specify pre-trained model or use the downloaded pretrained model specific in .yaml file
        language_pretrained_model_path = None        
        if config.NETWORK.BERT_PRETRAINED != '':
            # language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
            #                                                           config.NETWORK.BERT_PRETRAINED_EPOCH)
            #FM edit: just use path of pretrained model
            language_pretrained_model_path = config.NETWORK.BERT_PRETRAINED
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path

        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBertForDistance(
            config.NETWORK.VLBERT,
            language_pretrained_model_path=None if config.NETWORK.VLBERT.from_scratch else language_pretrained_model_path,
            with_rel_head=config.NETWORK.WITH_REL_LOSS,
            with_mlm_head=config.NETWORK.WITH_MLM_LOSS,
            with_mvrc_head=config.NETWORK.WITH_MVRC_LOSS,
        )

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED or (not self.config.NETWORK.MASK_RAW_PIXELS):
            self.object_mask_visual_embedding.weight.data.fill_(0.0)
        if self.config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding.weight.data.normal_(mean=0.0,
                                                                std=self.config.NETWORK.VLBERT.initializer_range)
        self.aux_text_visual_embedding.weight.data.normal_(mean=0.0, std=self.config.NETWORK.VLBERT.initializer_range)
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)

    def train(self, mode=True):
        super(ResNetVLBERTDistanceTranslationNoVision, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def forward(self,
                text,
                relationship_label,
                mlm_labels):

  
        ###########################################

        # prepare text
        text_input_ids = text
        # creates a text_tags tensor of the same shape as text tensor
        text_tags = text.new_zeros(text.shape)
        # ***** FM edit: blank out visual embeddings for translation retrieval task
        text_visual_embeddings = text_input_ids.new_zeros((text_input_ids.shape[0], text_input_ids.shape[1], 768), dtype=torch.float)
        text_visual_embeddings[:] = self.aux_text_visual_embedding.weight[0]

        # ****** FM edit: blank visual embeddings (use known dimensions)
        object_vl_embeddings = text_input_ids.new_zeros((text_input_ids.shape[0], 1, 1536), dtype=torch.float)

        # FM edit: No auxiliary text is used for text only
        # add auxiliary text - Concatenates the batches from the two dataloaders
        # The visual features for the text only corpus is just the embedding of the aux_visual_embedding (only one embedding)
        max_text_len = text_input_ids.shape[1]
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = (text_input_ids > 0)
        #FM: Edit: set to zero to ignore vision
        box_mask = text_input_ids.new_zeros((text_input_ids.shape[0], 1), dtype=torch.uint8)
        

        ###########################################

        # Visual Linguistic BERT

        relationship_logits_multi, mlm_logits_multi, mvrc_logits_multi, pooled_rep, text_out = self.vlbert(text_input_ids,
                                                                                     text_token_type_ids,
                                                                                     text_visual_embeddings,
                                                                                     text_mask,
                                                                                     object_vl_embeddings,
                                                                                     box_mask)

        ###########################################
        outputs = {}



        # FM edit: removed other two losses that are not defined
        outputs.update({
            'cls_output': text_out[:, 0, :]
        })

        # FM edit: losses not defined here - only used for inference
        loss = 0
        
        return outputs, loss
