export WANDB_DISABLED=true

# dataset=chess100k_gold_s_asa_n4
dataset=chess10k_gold_s_asa

exp=output/$dataset/DDM-s_asa-bs1024-lr3e-4-ep200-T20-`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
src/train_bash.py \
    --stage ddm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path chess_config \
    --do_train \
    --dataset $dataset \
    --finetuning_type full \
    --cutoff_len 328 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 3e-4 \
    --num_train_epochs 200.0 \
    --plot_loss \
    --run_name ${dataset}_prefix \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --time_reweighting linear \
    --topk_decoding True \
    > $exp/train.log


CUDA_VISIBLE_DEVICES=1  \
python3 -u src/train_bash.py \
    --stage ddm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path chess_config \
    --do_predict \
    --cutoff_len 328 \
    --dataset chess_test \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding True \
    > $exp/${dataset}/eval.log
done