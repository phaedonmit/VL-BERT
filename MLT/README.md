# MM-BERT additions


The code in this directory is used to train and test on the MLT task. Part of the code build upon the VL-BERT implementation 
but significant changes have been made specifically for the MLT task and the MM-BERT model. These are marked appropriately
with comments in the code marked as "FM edit" or "FM added". The MLT task can be used for training or testing as explained
in the README file of MM-BERT. Custom evaluation scripts have also been developed to obtain accuracy statistics after
the testing. These are:

1. evaluate.py
2. evaluate2018.py
3. compare_evaluate (for comparing testing results of different models)


Note: According to the options selected in the yaml file, the MLT task can be performed with or without the visual modality. 
These can be selected with the following options:

- with vision set:  "MODULE: ResNetVLBERTForPretrainingMultitask"
- without vision set:  "MODULE: ResNetVLBERTForPretrainingMultitaskNoVision"

