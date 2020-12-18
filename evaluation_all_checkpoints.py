import subprocess
import sys
import os

# model = "/data/faidon/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_LR6_fine_tune_global_lorikeet_single_phase/train_train/vl-bert_base_res101_pretrain_multitask-0003.model"
#model = "/data/faidon/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_LR6_fine_tune_global_lorikeet_two_phase/train_train/vl-bert_base_res101_pretrain_multitask-0003.model"
# location = "/data/faidon/VL-BERT/checkpoints/generated/single_phase_epoch03/"

model = sys.argv[1]
location = sys.argv[2]
model = "single_phase"


model_dir = "/data/faidon/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_LR6_fine_tune_global_lorikeet_"+model+"/train_train/"
location_dir = "/data/faidon/VL-BERT/checkpoints/generated/"

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
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_multi30k_FRDEmulti30k.yaml"
]

for file in os.listdir(model_dir):
    if file.endswith(".model"):
        print(os.path.join(model_dir, file))
exit()

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