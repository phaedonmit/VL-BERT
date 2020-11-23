DIRECTORY=$1
#root_path="/data/faidon/VL-BERT/checkpoints/generated/single_phase_epoch04/"
# root_path="/data/faidon/VL-BERT/checkpoints/generated/two_phase_epoch04/"
root_path="/data/faidon/VL-BERT/checkpoints/generated/$DIRECTORY/results.all.tok"
new_path="/data/faidon/VL-BERT/checkpoints/generated/$DIRECTORY/results_clean.all.tok"

sed '/*/d' < $root_path > $new_path
sed -i 's/|//g' $new_path
sed -i '/[A-Z]/d' $new_path
