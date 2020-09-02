######
# Author: Faidon Mitzalis
# Date: August 2020
# Comments: Save results to txt files
######
import os
import json

filepath = "/experiments/faidon/VL-BERT/checkpoints/output/pretrain/MLT/002_MLT_ende_no_vision/002_MLT_ende_no_vision_MLT_test2018.json"
targetpath = "/experiments/faidon/MLT_results"
model = filepath.split('/')[-1].split('.')[0]

target_file_predicted = os.path.join(targetpath, model+'_predictions')
target_file_ground_truths = os.path.join(targetpath, model+'_ground_truths')

predicted = []
ground_truth = []

with open(filepath) as json_file:
    data = json.load(json_file)

    #******************************************************
    # Step 1: Read json to dictionaries
    
    # json file to nested dictionary (each caption with all images)
    for p in data:
        predicted.append(p['word_pred'])
        ground_truth.append(p['word_de'])

with open(target_file_predicted, "w") as txt_file:
    for line in predicted:
        txt_file.write((line) + "\n") # works with any number of elements in a line

# with open(target_file_ground_truths, "w") as txt_file:
#     for line in ground_truth:
#         txt_file.write((line) + "\n") # works with any number of elements in a line


