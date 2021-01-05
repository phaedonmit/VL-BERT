echo 'Start evaluation for all...'
root_path=/data/faidon/VL-BERT/checkpoints/generated/

for d in checkpoints/generated/single_phase_epoch[0-1][0-9]/ ; do
    ./nmtpy_val_all_tok.sh d > "${root_path}${d}results_val.all.tok"
    echo "${root_path}${d}results_val.all.tok"
done