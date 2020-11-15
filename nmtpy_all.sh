root_path="/data/faidon/VL-BERT/checkpoints/generated/single_phase_epoch2/"
ground_path="/data/faidon/VL-BERT/data/ground_truth/"

echo "***************"
echo "1. DEIMG"
nmtpy-coco-metrics -l de "${root_path}DEIMG.txt" -r "${ground_path}DEIMG_0.txt" "${ground_path}DEIMG_1.txt" "${ground_path}DEIMG_2.txt" "${ground_path}DEIMG_3.txt" "${ground_path}DEIMG_4.txt"
