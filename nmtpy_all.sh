root_path="/data/faidon/VL-BERT/checkpoints/generated/single_phase_epoch2/"
ground_path="/data/faidon/VL-BERT/data/ground_truth/"

echo "***************"
echo "1. DEIMG"
nmtpy-coco-metrics -l de "${root_path}DEIMG.txt" -r "${ground_path}DEIMG_0.txt" "${ground_path}DEIMG_1.txt" "${ground_path}DEIMG_2.txt" "${ground_path}DEIMG_3.txt" "${ground_path}DEIMG_4.txt"
echo "***************"
echo "2. ENIMG"
nmtpy-coco-metrics -l en "${root_path}ENIMG.txt" -r "${ground_path}ENIMG_0.txt" "${ground_path}ENIMG_1.txt" "${ground_path}ENIMG_2.txt" "${ground_path}ENIMG_3.txt" "${ground_path}ENIMG_4.txt"
echo "***************"
echo "3. TUIMG"
nmtpy-coco-metrics -l tr "${root_path}TUIMG.txt" -r "${ground_path}TUIMG_0.txt" "${ground_path}TUIMG_1.txt"
echo "***************"
echo "4. DEENIMG"
nmtpy-coco-metrics -l en "${root_path}DEENIMG.txt" -r "${ground_path}DEENIMG.txt"
echo "***************"
echo "5. ENDEIMG"
nmtpy-coco-metrics -l de "${root_path}ENDEIMG.txt" -r "${ground_path}ENDEIMG.txt"
echo "***************"
echo "6. ENFRIMG"
nmtpy-coco-metrics -l fr "${root_path}ENFRIMG.txt" -r "${ground_path}ENFRIMG.txt"
echo "***************"
echo "7. FRENIMG"
nmtpy-coco-metrics -l en "${root_path}FRENIMG.txt" -r "${ground_path}FRENIMG.txt"
echo "***************"
echo "8. DEEN"
nmtpy-coco-metrics -l en "${root_path}DEEN.txt" -r "${ground_path}DEEN.txt"
echo "***************"
echo "9. ENDE"
nmtpy-coco-metrics -l de "${root_path}ENDE.txt" -r "${ground_path}ENDE.txt"
echo "***************"
echo "10. ENFR"
nmtpy-coco-metrics -l fr "${root_path}ENFR.txt" -r "${ground_path}ENFR.txt"
echo "***************"
echo "11. ENTU"
nmtpy-coco-metrics -l tr "${root_path}ENTU.txt" -r "${ground_path}ENTU.txt"
echo "***************"
echo "12. FREN"
nmtpy-coco-metrics -l en "${root_path}FREN.txt" -r "${ground_path}FREN.txt"
echo "***************"
echo "13. TUEN"
nmtpy-coco-metrics -l en "${root_path}TUEN.txt" -r "${ground_path}TUEN.txt"
