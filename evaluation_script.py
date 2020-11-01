import subprocess

model = "/experiments/faidon/test/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_LR6_fine_tune_global_durga_ALL/train_train/vl-bert_base_res101_pretrain_multitask-0000.model"

cfgs = [
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_DEIMG.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_ENIMG.yaml", 
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_FRIMG.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_TUIMG.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_DEENIMG.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_ENDEIMG.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_ENFRIMG.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_FRENIMG.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_DEEN.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_ENDE.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_ENFR.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_ENTU.yaml",
        # "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_FREN.yaml",
        "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_TUEN.yaml"
        ]

splits = [
        # "test2015",
        # "test2015",
        # "test_2016_fr",
        # "test_tu",
        # "test2015",
        # "test2015",
        # "test_2016_fr",
        # "test_2016_fr",
        # "test2015",
        # "test2015",
        # "test_2016_fr",
        # "test_tu",
        # "test_2016_fr",
        "test_tu"
]

lang = [
        # 'second', 
        # "first", 
        # "second", 
        # "second", 
        # "first", 
        # "second", 
        # "second",
        # "first", 
        # "first",
        # "second",
        # "second", 
        # "second", 
        # "first", 
        "first"
        ]



# cfgs = [
#         "/experiments/faidon/test/VL-BERT/cfgs/global_generate/base_prec_16x16G_fp16_MT_LR6_global_generate_image_only_TUIMG.yaml",

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
                                "--result-path", "/experiments/faidon/test/VL-BERT/checkpoints/generated",
                                "--result-name", name,
                                "--split", splits[i]                           
                                ])
    print("The exit code was: %d" % list_files.returncode)

#     list_files = subprocess.run(["python", "MT/evaluate_translation.py", name, 
#                                 "/experiments/faidon/test/VL-BERT/checkpoints/generated/"+name+".json",
#                                 lang[i]
#                                 ])
#     print("The exit code was: %d" % list_files.returncode)


