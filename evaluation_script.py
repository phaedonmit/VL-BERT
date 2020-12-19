import subprocess
import sys


# model = "/data/faidon/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_LR6_fine_tune_global_lorikeet_single_phase/train_train/vl-bert_base_res101_pretrain_multitask-0003.model"
#model = "/data/faidon/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_LR6_fine_tune_global_lorikeet_two_phase/train_train/vl-bert_base_res101_pretrain_multitask-0003.model"
model = sys.argv[1]
location = sys.argv[2]
# location = "/data/faidon/VL-BERT/checkpoints/generated/single_phase_epoch03/"

# cfgs = [
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_DEIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_ENIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_FRIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_TUIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_DEENIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_ENDEIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_ENFRIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_FRENIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_DEEN.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_ENDE.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_ENFR.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_ENTU.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_FREN.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_TUEN.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_DEFRIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_FRDEIMG.yaml",
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_FRIMG.yaml",
#         ]

splits = [
    "test2015",
    "test2015",
    # "test_2016_fr",
    "test_tu",
    "test2015",
    "test2015",
    "test_2016_fr",
    "test_2016_fr",
    "test2015",
    "test2015",
    "test_2016_fr",
    "test_tu",
    "test_2016_fr",
    "test_tu"
]
cfgs = [
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_DEIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_ENIMG.yaml",
    # "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_FRIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_TUIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_DEENIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_ENDEIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_ENFRIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_FRENIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_DEEN.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_ENDE.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_ENFR.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_ENTU.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_FREN.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_TUEN.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_DEFRIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_FRDEIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_FRIMG.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_multi30k_DEENmulti30k.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_multi30k_ENDEmulti30k.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_multi30k_ENFRmulti30k.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_multi30k_FRENmulti30k.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_multi30k_DEFRmulti30k.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_multi30k_FRDEmulti30k.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_DEFRWMT.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_FRDEWMT.yaml"
]


# lang = [
#         'second',
#         "first",
#         # "second",
#         "second",
#         "first",
#         "second",
#         "second",
#         "first",
#         "first",
#         "second",
#         "second",
#         "second",
#         "first",
#         "first",

#         ]


# cfgs = [
#         "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_TUIMG.yaml",


#         ]
# splits = ['test_tu']

# lang = ['second']
# cfgs = [
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_FRIMG.yaml",

#         ]
# splits = ['test_2016_fr']

# lang = ['second']


for i in range(len(cfgs)):
    name = cfgs[i].split("/")[-1].split('.')[0].split("_")[-1]

    print('***************')
    print(name)
    list_files = subprocess.run(["python", "MT/test.py",
                                 "--cfg", cfgs[i],
                                 "--ckpt", model,
                                 "--gpus", "0",
                                 "--result-path", location,
                                 "--result-name", name,
                                 # "--split", splits[i]
                                 "--split", 'test2015'
                                 ])
    print("The exit code was: %d" % list_files.returncode)

#     list_files = subprocess.run(["python", "MT/evaluate_translation.py", name,
#                                 "/experiments/faidon/test/VL-BERT/checkpoints/generated/"+name+".json",
#                                 lang[i]
#                                 ])
#     print("The exit code was: %d" % list_files.returncode)
