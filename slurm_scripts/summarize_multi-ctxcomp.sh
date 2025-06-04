#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=mdsumm
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/neuclir-irlab-ams

# Start the experiment.
MODEL_DIR=/ivi/ilps/personal/dju/checkpoints
TYPE=ctxcomp-v2-flan-t5-large-inverted-mds-std-310
for split in dev test;do
    python3 multidoc_summarize.py \
        --model_name_or_path ${MODEL_DIR}/${TYPE}/checkpoint-10000 \
        --model_class fid \
        --max_length 1024 \
        --eval_file data/requests-${split}-all.jsonl \
        --candidate_jsonl data/hits-${split}-all-summary.jsonl \
        --output_key multidoc-summary \
        --output_file data/hits-${split}-all-mdsummary-type0.jsonl
done


TYPE=ctxcomp-v2-flan-t5-large-inverted-mds-std-311
for split in dev test;do
    python3 multidoc_summarize.py \
        --model_name_or_path ${MODEL_DIR}/${TYPE}/checkpoint-10000 \
        --model_class fid \
        --max_length 1024 \
        --eval_file data/requests-${split}-all.jsonl \
        --candidate_jsonl data/hits-${split}-all-summary.jsonl \
        --output_key multidoc-summary \
        --output_file data/hits-${split}-all-mdsummary-type1.jsonl
done

