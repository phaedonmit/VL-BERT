echo 'Start cleaning for all...'
DIRECTORY=$1

for d in checkpoints/generated/${DIRECTORY}_epoch0[0-9]/ ; do
    root_path="/data/faidon/VL-BERT/${d}results_val.all.tok"
    new_path="/data/faidon/VL-BERT/${d}results_clean_val.all.tok"

    sed '/*/d' < $root_path > $new_path
    sed -i 's/|//g' $new_path
    sed -i '/[A-Z]/d' $new_path
    sed -i 's/ \+/ /g' $new_path
    echo $d
    cat $new_path
done


