python finetune_kk.py \
        --train_data "train/people3_num1000.jsonl" \
        --test_data  "test/people3_num100.jsonl" \
        --run_name kk_ft_ppl3_cot \
        --output_dir ./result/out_cot/train3  \
        --cot_ft \
        --num_train_epochs 50 \
        --save_strategy steps \
        --save_steps 0.2 \
        --max_seq_length 512 \
        --eval_steps 5 \
