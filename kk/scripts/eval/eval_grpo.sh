config="vllm"
max_token=4096
num_limit=100

echo "Processing num_limit: $num_limit"

export HF_ENDPOINT=https://hf-mirror.com
export TP_SIZE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn


models=(
    "./checkpoints/GRPO_logic_KK/Qwen-7B-bsz128-s2-1/global_step_380/actor/huggingface"
)
for model in "${models[@]}"; 
do
    echo "Processing model: $model"    
    python eval_kk.py --batch_size 100 --model ${model} --max_token ${max_token} --arch ${model} --ntrain 0 --config ${config} --use_vllm  --limit ${num_limit}  --split "test" --mode "zero-unified"
done
