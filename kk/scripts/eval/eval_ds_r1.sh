config="vllm"
max_token=15000
num_limit=100

echo "Processing num_limit: $num_limit"

export HF_ENDPOINT=https://hf-mirror.com
export TP_SIZE=4
export MAX_SEQ_LEN=16000
export VLLM_WORKER_MULTIPROC_METHOD=spawn

models=(
    /path/to/model/
)
for model in "${models[@]}"; 
do
    echo "Processing model: $model"    
    python eval_kk.py --batch_size 100 --model ${model} --max_token ${max_token} --arch ${model} --ntrain 0 --config ${config} --use_vllm  --limit ${num_limit}  --split "test" --mode "normal" --temperature 0.6
done
