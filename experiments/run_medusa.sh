USE_LINEAR=1 USE_LOCAL=1 CUDA_VISIBLE_DEVICES=0 python gen_model_answer_medusa.py  --model-path ./llama-2-7b-medusa-head-local-linear --base-model ./llama-2-7b-local-linear --model-id medusa-alpaca-local

USE_LINEAR=1 USE_LOCAL=1 CUDA_VISIBLE_DEVICES=0 python gen_model_answer_baseline.py  --model-path ./llama-2-7b-medusa-head-local-linear --base-model ./llama-2-7b-local-linear --model-id medusa-alpaca-local



USE_LINEAR=1 USE_LOCAL=1 USE_GLOBAL=1 GLOBAL_FACTOR=1 CUDA_VISIBLE_DEVICES="0"  python gen_model_answer_medusa.py  --model-path  ./llama-2-7b-medusa-head-grouped-linear --base-model ./llama-2-7b-grouped-linear --model-id medusa-alpaca-grouped

USE_LINEAR=1 USE_LOCAL=1 USE_GLOBAL=1 GLOBAL_FACTOR=1 CUDA_VISIBLE_DEVICES="0"  python gen_model_answer_baseline.py  --model-path  ./llama-2-7b-medusa-head-grouped-linear --base-model ./llama-2-7b-grouped-linear --model-id medusa-alpaca-grouped


USE_LINEAR=1 USE_LOCAL=1 USE_GLOBAL=1 GLOBAL_FACTOR=1 ADD_CONV=1 CUDA_VISIBLE_DEVICES="0"  python gen_model_answer_medusa.py  --model-path ./llama-2-7b-medusa-head-aug-linear --base-model ./llama-2-7b-aug-linear/ --model-id medusa-alpaca-aug

USE_LINEAR=1 USE_LOCAL=1 USE_GLOBAL=1 GLOBAL_FACTOR=1 ADD_CONV=1 CUDA_VISIBLE_DEVICES="0"  python gen_model_answer_baseline.py  --model-path ./llama-2-7b-medusa-head-aug-linear  --base-model ./llama-2-7b-aug-linear/ --model-id medusa-alpaca-aug