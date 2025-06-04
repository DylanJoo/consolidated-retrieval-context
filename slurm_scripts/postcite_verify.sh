#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=phc-v
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=10:00:00
#SBATCH --output=logs/neuclir-%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/neuclir-irlab-ams

for split in dev test;do
    echo finalize result for postcite neuclir-$split-all
    python3 tools/plaidx_postcite_verify.py \
        --report_json data/gptqa-${split}-all.json \
        --run_id irlab-ams-postcite-v \
        --submission data/submussion-${split}-all-irlab-ams-postcite-v \
        --batch_size 64 \
        --max_length 512 \
        --max_word_length 50 \
        --top_k 30 \
        --fact_threshold 0.9
    echo -e
    echo -e
done


