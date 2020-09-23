import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBertForPretraining
from common.utils.misc import soft_cross_entropy

BERT_WEIGHTS_NAME = 'pytorch_model.bin'



class ResNetVLBERTForPretrainingMultitask(Module):
    def __init__(self, config):

        super(ResNetVLBERTForPretrainingMultitask, self).__init__(config)

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

        self.vlbert = VisualLinguisticBertForPretraining(
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
        super(ResNetVLBERTForPretrainingMultitask, self).train(mode)
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
                image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels,
                image2,
                boxes2,
                im_info2,
                text2,
                relationship_label2,
                mlm_labels2,
                mvrc_ops2,
                mvrc_labels2,
                image3,
                boxes3,
                im_info3,
                text3,
                relationship_label3,
                mlm_labels3,
                mvrc_ops3,
                mvrc_labels3):
                # *aux):

        # concat aux texts from different dataset
        # assert len(aux) > 0 and len(aux) % 2 == 0
        # print('*******************')
        # print('* Second dataset: ')
        # print(boxes.shape)
        # print(boxes2.shape)
        # print(boxes3.shape)
        # exit()
        # print('************')
        # print('labels: ')
        # print(mlm_labels)
        # print(mlm_labels2)
        # print(mlm_labels3)
        # exit()

        ###########################################
        #### VISUAL FEATURE EXTRACTION

        # Get max_len for both
        images = image
        box_mask = (boxes[:, :, 0] > -1.5)
        origin_len = boxes.shape[1]

        images2 = image2
        box_mask2 = (boxes2[:, :, 0] > -1.5)
        origin_len2 = boxes2.shape[1]

        images3 = image3
        box_mask3 = (boxes3[:, :, 0] > -1.5)
        origin_len3 = boxes3.shape[1]
        
        max_len = max([int(box_mask.sum(1).max().item()), int(box_mask2.sum(1).max().item()), int(box_mask3.sum(1).max().item())])

        # Dataset 1: Visual feature extraction
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        mvrc_ops = mvrc_ops[:, :max_len]
        mvrc_labels = mvrc_labels[:, :max_len]

        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED:
            box_features = boxes[:, :, 4:]
            box_features[mvrc_ops == 1] = self.object_mask_visual_embedding.weight[0]
            boxes[:, :, 4:] = box_features

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None,
                                                mvrc_ops=mvrc_ops,
                                                mask_visual_embed=self.object_mask_visual_embedding.weight[0]
                                                if (not self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED)
                                                   and (not self.config.NETWORK.MASK_RAW_PIXELS)
                                                else None)

        # Dataset 2: Visual feature extraction

        box_mask2 = box_mask2[:, :max_len]
        boxes2 = boxes2[:, :max_len]
        mvrc_ops2 = mvrc_ops2[:, :max_len]
        mvrc_labels2 = mvrc_labels2[:, :max_len]

        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED:
            box_features2 = boxes2[:, :, 4:]
            box_features2[mvrc_ops == 1] = self.object_mask_visual_embedding.weight[0]
            boxes2[:, :, 4:] = box_features2

        obj_reps2 = self.image_feature_extractor(images=images2,
                                                boxes=boxes2,
                                                box_mask=box_mask2,
                                                im_info=im_info2,
                                                classes=None,
                                                segms=None,
                                                mvrc_ops=mvrc_ops2,
                                                mask_visual_embed=self.object_mask_visual_embedding.weight[0]
                                                if (not self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED)
                                                   and (not self.config.NETWORK.MASK_RAW_PIXELS)
                                                else None)

        # Dataset 3: Visual feature extraction

        box_mask3 = box_mask3[:, :max_len]
        boxes3 = boxes3[:, :max_len]
        mvrc_ops3 = mvrc_ops3[:, :max_len]
        mvrc_labels3 = mvrc_labels3[:, :max_len]

        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED:
            box_features3 = boxes3[:, :, 4:]
            box_features3[mvrc_ops == 1] = self.object_mask_visual_embedding.weight[0]
            boxes3[:, :, 4:] = box_features3

        obj_reps3 = self.image_feature_extractor(images=images3,
                                                boxes=boxes3,
                                                box_mask=box_mask3,
                                                im_info=im_info3,
                                                classes=None,
                                                segms=None,
                                                mvrc_ops=mvrc_ops3,
                                                mask_visual_embed=self.object_mask_visual_embedding.weight[0]
                                                if (not self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED)
                                                   and (not self.config.NETWORK.MASK_RAW_PIXELS)
                                                else None)

        # print("*********")
        # print("Visual features for both: ")
        # print(obj_reps['obj_reps_raw'].shape)
        # print(obj_reps2['obj_reps_raw'].shape)
        # print(obj_reps3['obj_reps_raw'].shape)
        # print('max length: ', max_len)

        # exit()
        ############################################
        ######### PREPARE TEXT

        # Dataset 1: Prepare text
        
        # prepare text
        text_input_ids = text
        # creates a text_tags tensor of the same shape as text tensor
        text_tags = text.new_zeros(text.shape)
        text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        # linguistic embedding for visual uses [IMG] embedding for all (apart from masked visual)
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        if self.config.NETWORK.WITH_MVRC_LOSS:
            object_linguistic_embeddings[mvrc_ops == 1] = self.object_mask_word_embedding.weight[0]
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        # Dataset 2: Prepare text

        # prepare text
        text_input_ids2 = text2
        # creates a text_tags tensor of the same shape as text tensor
        text_tags2 = text2.new_zeros(text2.shape)
        text_visual_embeddings2 = self._collect_obj_reps(text_tags2, obj_reps2['obj_reps'])

        # linguistic embedding for visual uses [IMG] embedding for all (apart from masked visual)
        object_linguistic_embeddings2 = self.object_linguistic_embeddings(
            boxes2.new_zeros((boxes2.shape[0], boxes2.shape[1])).long()
        )
        if self.config.NETWORK.WITH_MVRC_LOSS:
            object_linguistic_embeddings2[mvrc_ops2 == 1] = self.object_mask_word_embedding.weight[0]

        object_vl_embeddings2 = torch.cat((obj_reps2['obj_reps'], object_linguistic_embeddings2), -1)

        # Dataset 3: Prepare text and combine [IMG] with visual 

        # prepare text
        text_input_ids3 = text3
        # creates a text_tags tensor of the same shape as text tensor
        text_tags3 = text3.new_zeros(text3.shape)
        text_visual_embeddings3 = self._collect_obj_reps(text_tags3, obj_reps3['obj_reps'])

        # linguistic embedding for visual uses [IMG] embedding for all (apart from masked visual)
        object_linguistic_embeddings3 = self.object_linguistic_embeddings(
            boxes3.new_zeros((boxes3.shape[0], boxes3.shape[1])).long()
        )
        if self.config.NETWORK.WITH_MVRC_LOSS:
            object_linguistic_embeddings3[mvrc_ops == 1] = self.object_mask_word_embedding.weight[0]
        object_vl_embeddings3 = torch.cat((obj_reps3['obj_reps'], object_linguistic_embeddings3), -1)

        # print('********')
        # print('visual Shapes: ')
        # print(object_linguistic_embeddings.shape)
        # print(object_linguistic_embeddings2.shape)
        # print(object_linguistic_embeddings3.shape)
        # print('********')
        # print('objectvl Shapes: ')
        # print(object_vl_embeddings.shape)
        # print(object_vl_embeddings2.shape)
        # print(object_vl_embeddings3.shape)
        # print('********')
        # print('text Shapes: ')
        # print(text_visual_embeddings.shape)
        # print(text_visual_embeddings2.shape)
        # print(text_visual_embeddings3.shape)
        # exit()


        ################################
        ########  CONCATENATE BATCHES FROM 3 DATALOADERS

        # add auxiliary text - Concatenates the batches from the two dataloaders
        # The visual features for the text only corpus is just the embedding of the aux_visual_embedding (only one embedding)
        max_text_len = max(text_input_ids.shape[1], text_input_ids2.shape[1], text_input_ids3.shape[1])
        sep_batch = text_input_ids.shape[0] + text_input_ids2.shape[0]
        total_batch = text_input_ids.shape[0] + text_input_ids2.shape[0] + text_input_ids3.shape[0]

        text_input_ids_multi = text_input_ids.new_zeros((total_batch, max_text_len))
        text_input_ids_multi[:text_input_ids.shape[0], :text_input_ids.shape[1]] = text_input_ids
        text_input_ids_multi[text_input_ids.shape[0]:sep_batch, :text_input_ids2.shape[1]] = text_input_ids2
        text_input_ids_multi[sep_batch:, :text_input_ids3.shape[1]] = text_input_ids3

        text_token_type_ids_multi = text_input_ids_multi.new_zeros(text_input_ids_multi.shape)
        text_mask_multi = (text_input_ids_multi > 0)

        text_visual_embeddings_multi = text_visual_embeddings.new_zeros((total_batch,
                                                                         max_text_len,
                                                                         text_visual_embeddings.shape[-1]))
        text_visual_embeddings_multi[:text_visual_embeddings.shape[0], :text_visual_embeddings.shape[1]] \
            = text_visual_embeddings
        text_visual_embeddings_multi[text_visual_embeddings.shape[0]:sep_batch, :text_visual_embeddings2.shape[1]] \
            = text_visual_embeddings2
        text_visual_embeddings_multi[sep_batch:, :text_visual_embeddings3.shape[1]] \
            = text_visual_embeddings3

        object_vl_embeddings_multi = object_vl_embeddings.new_zeros((total_batch,
                                                                     *object_vl_embeddings.shape[1:]))
        object_vl_embeddings_multi[:object_vl_embeddings.shape[0]] = object_vl_embeddings
        object_vl_embeddings_multi[object_vl_embeddings.shape[0]:sep_batch] = object_vl_embeddings2
        object_vl_embeddings_multi[sep_batch:] = object_vl_embeddings3


        box_mask_multi = box_mask.new_zeros((total_batch, *box_mask.shape[1:]))
        box_mask_multi[:box_mask.shape[0]] = box_mask
        box_mask_multi[box_mask.shape[0]:sep_batch] = box_mask2
        box_mask_multi[sep_batch:] = box_mask3

        ###########################################

        # # Visual Linguistic BERT
        # print('text input shape: ', text)
        # print( 'text_input_ids_multi shape: ', text_input_ids_multi.shape)
        # print( 'text_token_type_ids_multi shape: ', text_token_type_ids_multi.shape)
        # print( 'text_visual_embeddings_multi shape: ', text_visual_embeddings_multi.shape)
        # print( 'text_mask_multi shape: ', text_mask_multi.shape)
        # print( 'object_vl_embeddings_multi shape: ', object_vl_embeddings_multi.shape)
        # print( 'box_mask_multi shape: ', box_mask_multi.shape)

        # print ('text_mask_multi: ', text_mask_multi)
        # print ('box_mask_multi: ', box_mask_multi)

        # exit()


        relationship_logits_multi, mlm_logits_multi, mvrc_logits_multi = self.vlbert(text_input_ids_multi,
                                                                                     text_token_type_ids_multi,
                                                                                     text_visual_embeddings_multi,
                                                                                     text_mask_multi,
                                                                                     object_vl_embeddings_multi,
                                                                                     box_mask_multi)

        # print('Logits: ')
        # print('logits shape: ', mlm_logits_multi.shape)
        # exit()

        ###########################################
        outputs = {}

        # loss
        relationship_loss = im_info.new_zeros(())
        mlm_loss = im_info.new_zeros(())
        mvrc_loss = im_info.new_zeros(())
        if self.config.NETWORK.WITH_REL_LOSS:
            relationship_logits = relationship_logits_multi[:text_input_ids.shape[0]]
            relationship_loss = F.cross_entropy(relationship_logits, relationship_label)
        if self.config.NETWORK.WITH_MLM_LOSS:
            mlm_labels_multi = mlm_labels.new_zeros((total_batch, max_text_len)).fill_(
                -1)
            mlm_labels_multi[:text_input_ids.shape[0], :mlm_labels.shape[1]] = mlm_labels
            mlm_labels_multi[text_input_ids.shape[0]:sep_batch, :mlm_labels2.shape[1]] = mlm_labels2
            mlm_labels_multi[sep_batch:, :mlm_labels3.shape[1]] = mlm_labels3

            mlm_logits_multi_padded = \
                mlm_logits_multi.new_zeros((*mlm_labels_multi.shape, mlm_logits_multi.shape[-1])).fill_(-10000.0)
            mlm_logits_multi_padded[:, :mlm_logits_multi.shape[1]] = mlm_logits_multi
            
            mlm_logits_multi = mlm_logits_multi_padded

            mlm_logits_dataset1 = mlm_logits_multi_padded[:text_input_ids.shape[0]]
            mlm_labels_dataset1 = mlm_labels_multi[:text_input_ids.shape[0]]

            mlm_logits_dataset2 = mlm_logits_multi_padded[text_input_ids.shape[0]:sep_batch]
            mlm_labels_dataset2 = mlm_labels_multi[text_input_ids.shape[0]:sep_batch]

            mlm_logits_dataset3 = mlm_logits_multi_padded[sep_batch:]
            mlm_labels_dataset3 = mlm_labels_multi[sep_batch:]


            if self.config.NETWORK.MLM_LOSS_NORM_IN_BATCH_FIRST:
                mlm_loss_dataset1 = F.cross_entropy(mlm_logits_dataset1.transpose(1, 2),
                                               mlm_labels_dataset1,
                                               ignore_index=-1, reduction='none')
                num_mlm_dataset1 = (mlm_labels_dataset1 != -1).sum(1, keepdim=True).to(dtype=mlm_loss_dataset1.dtype)
                num_has_mlm_dataset1 = (num_mlm_dataset1 != 0).sum().to(dtype=mlm_loss_dataset1.dtype)
                mlm_loss_dataset1 = (mlm_loss_dataset1 / (num_mlm_dataset1 + 1e-4)).sum() / (num_has_mlm_dataset1 + 1e-4)
                
                mlm_loss_dataset2 = F.cross_entropy(mlm_logits_dataset2.transpose(1, 2),
                                               mlm_labels_dataset2,
                                               ignore_index=-1, reduction='none')
                num_mlm_dataset2 = (mlm_labels_dataset2 != -1).sum(1, keepdim=True).to(dtype=mlm_loss_dataset2.dtype)
                num_has_mlm_dataset2 = (num_mlm_dataset2 != 0).sum().to(dtype=mlm_loss_dataset2.dtype)
                mlm_loss_dataset2 = (mlm_loss_dataset1 / (num_mlm_dataset1 + 1e-4)).sum() / (num_has_mlm_dataset1 + 1e-4)
                
                mlm_loss_dataset3 = F.cross_entropy(mlm_logits_dataset3.transpose(1, 2),
                                               mlm_labels_dataset3,
                                               ignore_index=-1, reduction='none')
                num_mlm_dataset3 = (mlm_labels_dataset3 != -1).sum(1, keepdim=True).to(dtype=mlm_loss_dataset3.dtype)
                num_has_mlm_dataset3 = (num_mlm_dataset3 != 0).sum().to(dtype=mlm_loss_dataset3.dtype)
                mlm_loss_dataset3 = (mlm_loss_dataset3 / (num_mlm_dataset3 + 1e-4)).sum() / (num_has_mlm_dataset3 + 1e-4)                
            else:
                # mlm_loss = F.cross_entropy(mlm_logits_multi_padded.view((-1, mlm_logits_multi_padded.shape[-1])),
                #                            mlm_labels_multi.view(-1),
                #                            ignore_index=-1)
                mlm_loss_dataset1 = F.cross_entropy(
                    mlm_logits_dataset1.view((-1, mlm_logits_multi_padded.shape[-1])),
                    mlm_labels_dataset1.view(-1),
                    ignore_index=-1
                )
                mlm_loss_dataset2 = F.cross_entropy(
                    mlm_logits_dataset3.view((-1, mlm_logits_multi_padded.shape[-1])),
                    mlm_labels_dataset3.view(-1),
                    ignore_index=-1
                )
                mlm_loss_dataset3 = F.cross_entropy(
                    mlm_logits_dataset3.view((-1, mlm_logits_multi_padded.shape[-1])),
                    mlm_labels_dataset3.view(-1),
                    ignore_index=-1
                )


        # mvrc_loss = F.cross_entropy(mvrc_logits.contiguous().view(-1, mvrc_logits.shape[-1]),
        #                             mvrc_labels.contiguous().view(-1),
        #                             ignore_index=-1)
        if self.config.NETWORK.WITH_MVRC_LOSS:
            mvrc_logits = mvrc_logits_multi[:mvrc_labels.shape[0], :mvrc_labels.shape[1]]
            if self.config.NETWORK.MVRC_LOSS_NORM_IN_BATCH_FIRST:
                mvrc_loss = soft_cross_entropy(
                    mvrc_logits.contiguous().view(-1, mvrc_logits.shape[-1]),
                    mvrc_labels.contiguous().view(-1, mvrc_logits.shape[-1]),
                    reduction='none').view(mvrc_logits.shape[:-1])
                valid = (mvrc_labels.sum(-1) - 1).abs() < 1.0e-1
                mvrc_loss = (mvrc_loss / (valid.sum(1, keepdim=True).to(dtype=mvrc_loss.dtype) + 1e-4)) \
                                .sum() / ((valid.sum(1) != 0).sum().to(dtype=mvrc_loss.dtype) + 1e-4)
            else:
                mvrc_loss = soft_cross_entropy(mvrc_logits.contiguous().view(-1, mvrc_logits.shape[-1]),
                                               mvrc_labels.contiguous().view(-1, mvrc_logits.shape[-1]))

            mvrc_logits_padded = mvrc_logits.new_zeros((mvrc_logits.shape[0], origin_len, mvrc_logits.shape[2])).fill_(
                -10000.0)
            mvrc_logits_padded[:, :mvrc_logits.shape[1]] = mvrc_logits
            mvrc_logits = mvrc_logits_padded
            mvrc_labels_padded = mvrc_labels.new_zeros((mvrc_labels.shape[0], origin_len, mvrc_labels.shape[2])).fill_(
                0.0)
            mvrc_labels_padded[:, :mvrc_labels.shape[1]] = mvrc_labels
            mvrc_labels = mvrc_labels_padded

        outputs.update({
            'relationship_logits': relationship_logits if self.config.NETWORK.WITH_REL_LOSS else None,
            'relationship_label': relationship_label if self.config.NETWORK.WITH_REL_LOSS else None,
            'mlm_logits_dataset1': mlm_logits_dataset1 if self.config.NETWORK.WITH_MLM_LOSS else None,
            'mlm_label_dataset1': mlm_labels_dataset1 if self.config.NETWORK.WITH_MLM_LOSS else None,
            'mlm_logits_dataset2': mlm_logits_dataset2 if self.config.NETWORK.WITH_MLM_LOSS else None,
            'mlm_label_dataset2': mlm_labels_dataset2 if self.config.NETWORK.WITH_MLM_LOSS else None,
            'mlm_logits_dataset3': mlm_logits_dataset3 if self.config.NETWORK.WITH_MLM_LOSS else None,
            'mlm_label_dataset3': mlm_labels_dataset3 if self.config.NETWORK.WITH_MLM_LOSS else None,
            'mvrc_logits': mvrc_logits if self.config.NETWORK.WITH_MVRC_LOSS else None,
            'mvrc_label': mvrc_labels if self.config.NETWORK.WITH_MVRC_LOSS else None,
            'relationship_loss': relationship_loss,
            'mlm_loss_dataset1': mlm_loss_dataset1,
            'mlm_loss_dataset2': mlm_loss_dataset2,
            'mlm_loss_dataset3': mlm_loss_dataset3,
            'mvrc_loss': mvrc_loss,
        })

        loss = relationship_loss.mean() + mlm_loss_dataset1.mean() + mlm_loss_dataset2.mean() + mlm_loss_dataset3.mean() + mvrc_loss.mean()


        return outputs, loss
