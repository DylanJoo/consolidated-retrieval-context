#!/bin/sh
#SBATCH --job-name=phc
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=1
#SBATCH --time=09:00:00
#SBATCH --output=logs/neuclir-%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/neuclir-irlab-ams

for split in dev test;do
    echo finalize result for postcite neuclir-$split-all
    python3 tools/plaidx_postcite.py \
        --report_json data/gptqa-${split}-all.json \
        --run_id irlab-ams-postcite \
        --submission submissions/submission-${split}-all-irlab-ams-postcite

    echo -e
    echo -e
done
