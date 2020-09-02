# MM-BERT additions


The code in this directory is used to train and test on the MT task. Part of the code build upon the VL-BERT implementation 
but significant changes have been made specifically for the MT task and the MM-BERT model. These are marked appropriately
with comments in the code marked as "FM edit" or "FM added". The MT task can be used for training or testing as explained
in the README file of MM-BERT. There are two proposed architectures for the MT task, the simpler architecture has a single
Linear layer that acts as a decoder. The other architecture has uses the Encoder-Decoder architecture that where the Encoder
is initiliased with MM-BERT and the Decoder is again a Transformer with Cross-Attention with the Encoder output embeddings.
Custom evaluation scripts have also been developed to obtain BLEU statistics after
the testing. These are:

1. evaluate_trasnlation.py


Note: According to the options selected in the yaml file, the MT task can be performed with or without the visual modality.
Different configurations are used during inference, where the generated tokens are used as inputs for subsequent steps to 
generate the translated sequence the different options can be selected on the yaml file as follows: 

1) Method 1: Simple Encoder
    - Training:
        - with vision, set:  "MODULE: ResNetVLBERTForPretraining"
        - without vision, set:  "MODULE: ResNetVLBERTForPretrainingNoVision"
    - Inference:
        - with vision, set:  "MODULE: ResNetVLBERTForPretrainingGenerate"
        - without vision, set:  "MODULE: ResNetVLBERTForPretrainingGenerateNoVision"
2) Method 2: Encoder-Decoder architecture
    - Training:
        - with vision, set:  "MODULE: ResNetVLBERTForPretraining"
    - Inference:
        - with vision, set:  "MODULE: ResNetVLBERTForPretrainingEncDec"
        - without vision, set:  "MODULE: ResNetVLBERTForPretrainingEncDecGenerate"


