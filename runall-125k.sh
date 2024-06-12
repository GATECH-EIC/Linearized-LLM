#local+grouped
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file accelerateconfig-g8.yaml --num_processes 8 train-hf.py --dataset_name wiki40b --model_name_or_path gpt2 --output_dir ./test-clm-both-125k \
--dataset_config en --per_device_train_batch_size 16  --tokenizer_name gpt2 --weight_decay 0.01 --learning_rate 7e-4 --gradient_accumulation_steps 2 --num_warmup_steps 10000 --num_train_epochs 15 \
--checkpointing_steps epoch 

#local+grouped+AUG(ours)
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file accelerateconfig-g8.yaml --num_processes 8 train-hf.py --dataset_name wiki40b --model_name_or_path gpt2 --use_conv --output_dir ./test-clm-both-conv-125k \
--dataset_config en --per_device_train_batch_size 16  --tokenizer_name gpt2 --weight_decay 0.01 --learning_rate 7e-4 --gradient_accumulation_steps 2 --num_warmup_steps 10000 --num_train_epochs 15 \
--checkpointing_steps epoch 

#local+AUG(ours)
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file accelerateconfig-g8.yaml --num_processes 8 train-hf.py --dataset_name wiki40b --model_name_or_path gpt2 --use_conv --output_dir ./test-clm-local-conv-125k \
--dataset_config en --per_device_train_batch_size 16  --tokenizer_name gpt2 --weight_decay 0.01 --learning_rate 7e-4  --gradient_accumulation_steps 2 --num_warmup_steps 10000 --no_grouped --num_train_epochs 15 \
--checkpointing_steps epoch

#square
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file accelerateconfig-g8.yaml --num_processes 4 train-hf.py --dataset_name wiki40b --model_name_or_path gpt2 --output_dir ./test-clm-square-125k \
--dataset_config en --per_device_train_batch_size 16  --tokenizer_name gpt2 --weight_decay 0.01 --learning_rate 7e-4 --gradient_accumulation_steps 2 --num_warmup_steps 10000 --square --num_train_epochs 15 \
--checkpointing_steps epoch 

#local only
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file accelerateconfig-g8.yaml --num_processes 4 train-hf.py --dataset_name wiki40b --model_name_or_path gpt2 --output_dir ./test-clm-local-125k \
--dataset_config en --per_device_train_batch_size 16  --tokenizer_name gpt2 --weight_decay 0.01 --learning_rate 7e-4  --gradient_accumulation_steps 2 --num_warmup_steps 10000 --no_grouped --num_train_epochs 15 \
--checkpointing_steps epoch 

#grouped only
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file accelerateconfig-g8.yaml --num_processes 4 train-hf.py --dataset_name wiki40b --model_name_or_path gpt2 --output_dir ./test-clm-grouped-125k \
--dataset_config en --per_device_train_batch_size 16  --tokenizer_name gpt2 --weight_decay 0.01 --learning_rate 7e-4  --gradient_accumulation_steps 2 --num_warmup_steps 10000 --no_local --num_train_epochs 15 \
--checkpointing_steps epoch 
