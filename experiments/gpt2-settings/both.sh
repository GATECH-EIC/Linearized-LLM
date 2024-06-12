export TASK_NAME=$TASK

export MODEL_PATH=gpt2 #clm-script-wikipedia/tmp/test-run_gpt2_wikipedia-li-YOSO.sh
#mlm-script-wikipedia/tmp/test-run_bert_wikipedia_yoso-8.sh/
#clm-script-wikipedia/tmp/test-run_gpt2_wikipedia-li-YOSO.sh/

export CONV=0
export FFNCONV=0
export LINEAR=0
export LOCAL_ATTN=0
export GLOBAL_ATTN=0
export BOTH_ATTN=1
export FLASH_LOCAL=0
export FLASH_GLOBAL=0
export NO_FLASH_INF=1
export GROUP_SIZE=64

mkdir -p ./tmp/$TASK_NAME/$MODEL_PATH/
echo "running; {$TASK_NAME/$MODEL_PATH/}"
python run_gpt2.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --max_length 256 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./tmp/$TASK_NAME/$MODEL_PATH