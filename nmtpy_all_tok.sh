echo 'Start evaluation...'
DIRECTORY=$1
root_path="/data/faidon/VL-BERT/checkpoints/generated/$DIRECTORY/"
ground_path="/data/faidon/VL-BERT/data/ground_truth"

# nmtpy-coco-metrics -l de "/data/faidon/VL-BERT/checkpoints/generated/single_phase_ENDEIMG_epoch18/ENDEIMG.txt" -r "/data/faidon/VL-BERT/data/ground_truth/ENDEIMG.txt.tok"

echo "***************"
echo "1. DEIMG"
nmtpy-coco-metrics -l de "${root_path}DEIMG.txt" -r ${ground_path}/DEIMG_*.txt.tok
echo "***************"
echo "2. ENIMG"
nmtpy-coco-metrics -l en "${root_path}ENIMG.txt" -r ${ground_path}/ENIMG_*.txt.tok
echo "***************"
echo "3. TUIMG"
nmtpy-coco-metrics -l en "${root_path}TUIMG.txt" -r ${ground_path}/TUIMG_*.txt.tok
echo "***************"
echo "4. DEENIMG"
nmtpy-coco-metrics -l en "${root_path}DEENIMG.txt" -r "${ground_path}/DEENIMG.txt.tok"
echo "***************"
echo "5. ENDEIMG"
nmtpy-coco-metrics -l de "${root_path}ENDEIMG.txt" -r "${ground_path}/ENDEIMG.txt.tok"
echo "***************"
echo "6. ENFRIMG"
nmtpy-coco-metrics -l fr "${root_path}ENFRIMG.txt" -r "${ground_path}/ENFRIMG.txt.tok"
echo "***************"
echo "7. FRENIMG"
nmtpy-coco-metrics -l en "${root_path}FRENIMG.txt" -r "${ground_path}/FRENIMG.txt.tok"
echo "***************"
echo "8. DEEN"
nmtpy-coco-metrics -l en "${root_path}DEEN.txt" -r "${ground_path}/DEEN.txt.tok"
echo "***************"
echo "9. ENDE"
nmtpy-coco-metrics -l de "${root_path}ENDE.txt" -r "${ground_path}/ENDE.txt.tok"
echo "***************"
echo "10. ENFR"
nmtpy-coco-metrics -l fr "${root_path}ENFR.txt" -r "${ground_path}/ENFR.txt.tok"
echo "***************"
echo "11. ENTU"
nmtpy-coco-metrics -l en "${root_path}ENTU.txt" -r "${ground_path}/ENTU.txt.tok"
echo "***************"
echo "12. FREN"
nmtpy-coco-metrics -l en "${root_path}FREN.txt" -r "${ground_path}/FREN.txt.tok"
echo "***************"
echo "13. TUEN"
nmtpy-coco-metrics -l en "${root_path}TUEN.txt" -r "${ground_path}/TUEN.txt.tok"
echo "***************"
echo "14. ZS FRIMG"
nmtpy-coco-metrics -l fr "${root_path}FRIMG.txt" -r "${ground_path}/ENFRIMG.txt.tok"
echo "***************"
echo "15. ZS DEFRIMG"
nmtpy-coco-metrics -l de "${root_path}DEFRIMG.txt" -r "${ground_path}/ENFRIMG.txt.tok"
echo "***************"
echo "16. ZS FRDEIMG"
nmtpy-coco-metrics -l de "${root_path}FRDEIMG.txt" -r "${ground_path}/ENDEIMG.txt.tok"
echo "***************"
echo "17. NMT DEEN multi30k"
nmtpy-coco-metrics -l en "${root_path}DEENmulti30k.txt" -r "${ground_path}/DEENIMG.txt.tok"
echo "***************"
echo "18. NMT ENDE multi30k"
nmtpy-coco-metrics -l de "${root_path}ENDEmulti30k.txt" -r "${ground_path}/ENDEIMG.txt.tok"
echo "***************"
echo "19. NMT ENFR multi30k"
nmtpy-coco-metrics -l fr "${root_path}ENFRmulti30k.txt" -r "${ground_path}/ENFRIMG.txt.tok"
echo "***************"
echo "20. NMT FREN multi30k"
nmtpy-coco-metrics -l en "${root_path}FRENmulti30k.txt" -r "${ground_path}/FRENIMG.txt.tok"
echo "***************"
echo "21. NMT ZS DEFR multi30k"
nmtpy-coco-metrics -l de "${root_path}DEFRmulti30k.txt" -r "${ground_path}/ENFRIMG.txt.tok"
echo "***************"
echo "22. NMT ZS FRDE multi30k"
nmtpy-coco-metrics -l de "${root_path}FRDEmulti30k.txt" -r "${ground_path}/ENDEIMG.txt.tok"