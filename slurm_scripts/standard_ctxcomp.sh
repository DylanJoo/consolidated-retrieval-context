#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=std
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/neuclir-irlab-ams

# Start the experiment.

# ctxcomp: recomp top 30
run_id=irlab-ams-std-recomp-llama-8B
# generate output from local
# for split in test dev; do
#     python standard.py \
#         --config configs/neuclir.llama3-8b-chat.std-summary.0shot.yaml \
#         --split ${split} \
#         --eval_file data/requests-${split}-all.jsonl \
#         --candidates_jsonl data/hits-${split}-all-summary.jsonl \
#         --ndoc 30 \
#         --tag std-summary-top30 \
#         --used_field recomp-nq-summary
# done
# for split in dev test; do
#     python tools/convert_output_to_submission.py  \
#         --report_json results/report-std-summary-top30-${split}-all.json  \
#         --run_id ${run_id} \
#         --submission submissions/submission-${split}-all-${run_id}
# done

## ctxcomp: multidoc 310 top 30 
run_id=irlab-ams-std-mdcomp-330-translate-llama-8B
## generate output from local
for split in test; do
    python standard.py \
        --config configs/neuclir.llama3-8b-chat.std-summary.0shot.yaml \
        --split ${split} \
        --eval_file data/requests-${split}-all.jsonl \
        --candidates_jsonl data/hits-${split}-all-mdsummary-type0.jsonl \
        --ndoc 30 \
        --tag std-mdsummary-top30 \
        --used_field multidoc-summary 
done
for split in test; do
    python tools/convert_output_to_submission.py  \
        --report_json results/report-std-mdsummary-top30-${split}-all.json  \
        --run_id ${run_id} \
        --submission submissions/submission-${split}-all-${run_id}
done

## ctxcomp: multidoc 311 top 30 
# run_id=irlab-ams-std-mdcomp-331-translate-llama-8B
# ## generate output from local
# for split in dev; do
#     python standard.py \
#         --config configs/neuclir.llama3-8b-chat.std-summary.0shot.yaml \
#         --split ${split} \
#         --eval_file data/requests-${split}-all.jsonl \
#         --candidates_jsonl data/hits-${split}-all-mdsummary-type1.jsonl \
#         --ndoc 30 \
#         --tag std-mdsummary-top30 \
#         --used_field multidoc-summary 
# done
# for split in dev; do
#     python tools/convert_output_to_submission.py  \
#         --report_json results/report-std-mdsummary-top30-${split}-all.json  \
#         --run_id ${run_id} \
#         --submission submissions/submission-${split}-all-${run_id}
# done
