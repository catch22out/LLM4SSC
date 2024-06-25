set -x

BASE_MODEL_PATH="/home/Data/xac/nas/models/models--codellama--CodeLlama-13b-Instruct-hf/snapshots/745795438019e47e4dad1347a0093e11deee4c68"  # the pretrained LLM weights
TEST_DATA_PATH="data4test.json"  # the dataset used for finetuning
FT_MODEL_PATH="/home/Data/xac/merge/llm_one4all/finetuning/model/checkpoint-80"
OUTPUT_PATH="output_test.json"
CUDA_VISIBLE_DEVICES=0,1,2,3 python predict.py \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --peft_output ${FT_MODEL_PATH} \
    --data_path ${TEST_DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --model_max_length 4096 \
    --max_length 4096
    