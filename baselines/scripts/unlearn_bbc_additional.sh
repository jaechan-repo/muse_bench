CORPUS='bbc'
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"

TARGET_DIR='swj0419/bbc-original_STEP0000080_5-31'
RETRAIN_DIR='swj0419/bbc-retrain_STEP0000040_5-31'
LLAMA_DIR='meta-llama/Llama-2-7b-hf'

MAX_LEN=2048
EPOCHS=20
LR='1e-5'
PER_DEVICE_BATCH_SIZE=4 # 8 GPUs


for algo in 'ga_klr' 'npo_gdr' 'npo_klr'; do
    python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "./ckpt/$CORPUS/$algo" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
        --resume_from_checkpoint
done
