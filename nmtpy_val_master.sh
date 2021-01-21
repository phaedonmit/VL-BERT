echo 'Start evaluation for all...'
root_path=/data/faidon/VL-BERT/
DIRECTORY=$1

for d in checkpoints/generated/${DIRECTORY}_epoch[1][3-7]/ ; do
    echo "${root_path}${d}results_val.all.tok"
    ./nmtpy_val_all_tok.sh $d > "${root_path}${d}results_val.all.tok"
done