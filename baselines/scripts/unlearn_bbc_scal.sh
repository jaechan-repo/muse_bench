CORPUS='bbc'
TARGET_DIR="swj0419/bbc-original_STEP0000080_5-31"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"
CRIT='scal'
MAX_LEN=2048
PER_DEVICE_BATCH_SIZE=4

# Iterative unlearning methods
LR='1e-5'
RETAIN="../data/$CORPUS/raw/retain1.txt"
STEP_ALGOS=('npo_klr')
EPOCHS=('10')
for i in ${!STEP_ALGOS[*]}; do 
    algo=${STEP_ALGOS[$i]}
    epoch=${EPOCHS[$i]}
    for k in '2' '3' '4'; do
        python unlearn.py \
            --algo $algo \
            --data_file "../data/$CORPUS/$CRIT/forget_$k.txt" \
            --out_dir "./ckpt/$CORPUS/$CRIT/$algo/fold=$k" \
            --epochs $epoch --lr $LR \
            --retain_data_file $RETAIN --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR --per_device_batch_size $PER_DEVICE_BATCH_SIZE --max_len $MAX_LEN
    done
done

# Task Vector, WHP
FT_EPOCHS=10
FT_LR='1e-6'
for k in '2' '3' '4'; do
    python unlearn.py \
        --algo $algo \
        --data_file "../data/$CORPUS/$CRIT/forget_$k.txt" \
        --out_dir "./ckpt/$CORPUS/$CRIT/tv/$k" \
        --epochs $FT_EPOCHS --lr $FT_LR \
        --retain_data_file $RETAIN --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR --per_device_batch_size $FT_PER_DEVICE_BATCH_SIZE --max_len $MAX_LEN
done
