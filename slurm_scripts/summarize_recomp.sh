#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=summ
#SBATCH --partition gpu
#SBATCH --gres=gpu:tesla_p40:1
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

for split in dev test;do
    python3 summarize.py \
        --model_name_or_path fangyuan/nq_abstractive_compressor \
        --model_class seq2seq \
        --template 'Question: {Q}\n Document: {P}\n Summary: ' \
        --batch_size 32 \
        --max_length 512 \
        --eval_file data/requests-${split}-all.jsonl \
        --eval_rewrite_file data/gptqr-${split}-all.json \
        --candidate_tsv data/hits-${split}-all-translate.tsv \
        --output_key recomp-nq-summary \
        --output_file data/hits-${split}-all-summary.jsonl \
        --truncate
done

