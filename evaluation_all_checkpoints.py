import subprocess
import sys
import os

# model = "/data/faidon/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_LR6_fine_tune_global_lorikeet_single_phase/train_train/vl-bert_base_res101_pretrain_multitask-0003.model"
#model = "/data/faidon/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_LR6_fine_tune_global_lorikeet_two_phase/train_train/vl-bert_base_res101_pretrain_multitask-0003.model"
# location = "/data/faidon/VL-BERT/checkpoints/generated/single_phase_epoch03/"

config_name = sys.argv[1]
location = sys.argv[2]
config_name = "single_phase_ENDEIMG"
# config_name = "single_phase_ENDEIMG_MBERTonly"
# config_name = "single_phase_ENDEIMG_scratch"
# config_name = "single_phase_ENDEIMG_VLBERTonly"


model_dir = "/data/faidon/VL-BERT/checkpoints/output/pretrain/vl-bert/base_prec_16x16G_fp16_LR6_fine_tune_global_lorikeet_" + \
    config_name+"/train_train/"
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
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_multi30k_FRDEmulti30k.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_DEFRWMT.yaml",
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_no_vision_FRDEWMT.yaml"
]

cfgs = [
    "/data/faidon/VL-BERT/cfgs/global_generate_lorikeet/base_prec_16x16G_fp16_MT_LR6_global_generate_MMT_ENDEIMG.yaml"
]

for file in os.listdir(model_dir):
    if file.endswith(".model"):
        model = os.path.join(model_dir, file)
        epoch = file.split('.')[0][-2:]
        location = location_dir + config_name + '_epoch' + epoch
        print("Model path: ", model)
        print("Location path: ", location)

        for i in range(len(cfgs)):
            name = cfgs[i].split("/")[-1].split('.')[0].split("_")[-1]

            print('***************')
            print(name)
            list_files = subprocess.run(["python", "MT/test.py",
                                         "--cfg", cfgs[i],
                                         "--ckpt", model,
                                         "--gpus", "0",
                                         "--result-path", location,
                                         "--result-name", name+'_val',
                                         # "--split", splits[i]
                                         "--split", 'val'
                                         ])
            print("The exit code was: %d" % list_files.returncode)
