# MM-BERT additions


The code in this directory is used to train and test for the various retrieval tasks. Part of the code builds upon the VL-BERT implementation 
but significant changes have been made specifically for the Retrieval tasks and the MM-BERT model. These are marked appropriately
with comments in the code marked as "FM edit" or "FM added". The Retrieval task can be used for training or testing as explained
in the README file of MM-BERT.

Custom evaluation scripts have also been developed to recall at position k R@k for standard retrieval ranking statistics.
These are the following:

1. evalute.py: For 'Image-based caption retrieval' and 'Caption-based image retrieval' tasks
2. evaluate_translation.py: For the 'Translation retrieval' task
3. evaluate_distance.py: For the 'Translation retrieval' task using cosine similarities


Note: According to the options selected in the yaml file, the different retrieval tasks can be selected: 

1) 'Image-based caption retrieval' and 'Caption-based image retrieval'
    - "MODULE: ResNetVLBERTForPretrainingMultitask"
    - "DATASET: flickr30k" (dataset can be "multi30k", "multi30k_5x", "multi30k_5x_mixed")
2) 'Translation Retrieval' with fine-tuned model
    - "MODULE: ResNetVLBERTForPretrainingMultitask"
    - "DATASET: translation_multi30k"
3) 'Translation Retrieval' with cosine similarity
    - "MODULE: ResNetVLBERTDistanceTranslationNoVision"
    - "DATASET: distance_translation_multi30k"


