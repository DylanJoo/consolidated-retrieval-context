#!/bin/sh
#SBATCH --job-name=encdec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate crc

cd ~/consolidated-retrieval-context

accelerate launch \
    --config_file configs/default_train.yaml \
    train.py \
    --model_name_or_path google/flan-t5-small \
    --tokenizer_name google/flan-t5-small \
    --config_name google/flan-t5-small  \
    --output_dir ${MODEL_DIR}/ctxcomp-flan-t5-small \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --train_file data/mds-5k-greedy-1.jsonl \
    --max_src_length 256 \
    --max_tgt_length 256 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --max_steps 10000 \
    --save_steps 1000 \
    --eval_steps 500 \
    --do_train --do_eval \
    --bf16 \
    --max_num_contexts 5 \
    --num_distractor_docs 2 \
    --num_redundant_docs 1 \
    --collator_type standard \
    --report_to wandb --run_name encdec \
