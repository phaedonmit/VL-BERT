echo 'Start cleaning for all...'

for d in checkpoints/generated/single_phase_epoch[0-1][0-9]/ ; do
    root_path="/data/faidon/VL-BERT${d}/results_val.all.tok"
    new_path="/data/faidon/VL-BERT${d}/results_clean_val.all.tok"

    sed '/*/d' < $root_path > $new_path
    sed -i 's/|//g' $new_path
    sed -i '/[A-Z]/d' $new_path
    sed -i 's/ \+/ /g' $new_path
    cat $new_path
done


