export TASK_NAME=$TASK

export MODEL_PATH=t5-base #clm-script-wikipedia/tmp/test-run_gpt2_wikipedia-li-YOSO.sh
#mlm-script-wikipedia/tmp/test-run_bert_wikipedia_yoso-8.sh/
#clm-script-wikipedia/tmp/test-run_gpt2_wikipedia-li-YOSO.sh/
export ENCODER=1
export CROSS=1
export DECODER=1

export LINEAR_HEADS=0
export CONV=0
export FFNCONV=0
export LINEAR=0
export LOCAL_ATTN=0
export GLOBAL_ATTN=1
export BOTH_ATTN=0
export FLASH_LOCAL=0
export FLASH=0
export FLASH_GLOBAL=0
export NO_FLASH_INF=1
export GROUP_SIZE=64

mkdir -p ./tmp/$TASK_NAME/$MODEL_PATH/
echo "running; {$TASK_NAME/$MODEL_PATH/}"
python run_t5.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --max_length 256 \
  --pad_to_max_length \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./tmp/$TASK_NAME/$MODEL_PATH