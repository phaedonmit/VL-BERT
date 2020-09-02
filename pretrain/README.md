# MM-BERT additions


The code in this directory is used to pretrain the MM-BERT models. Part of the code build upon the VL-BERT implementation 
but significant changes have been made specifically for training the multilingual MM-BERT variants. The changes are marked appropriately
with comments in the code marked as "FM edit" or "FM added". The pretrain task can be used ran as explained
in the README file of MM-BERT. There three proposed pre-training configurations, Task A, Task B and Task C. These can be selected
by selecting the appropriate settings on the yaml file:


1) Task A: 3-way parallel data (Multi30k dataset). Set:
    - "MODULE:ResNetVLBERTForPretrainingMultitask"
    - DATASET: multi30k
    - INITIALISATION: 'hybrid'
2) Task B: 2x2-way parallel data (Multi30k dataset). Set:
    - "MODULE:ResNetVLBERTForPretrainingMultitask"
    - DATASET: multi30k_taskB
    - INITIALISATION: 'hybrid'
3) Task B: 2-way parallel data language only (WMT'14 dataset). Set:
    - "MODULE:ResNetVLBERTForPretrainingMultitask"
    - DATASET: ResNetVLBERTForPretrainingTaskC
    - INITIALISATION: 'hybrid'

