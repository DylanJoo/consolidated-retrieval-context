cd /home/dju/neuclir-irlab-ams/

for split in dev test;do
    python3 tools/prepare_gpt_answer.py \
        --input_jsonl data/neuclir/requests-${split}-all.jsonl \
        --output_csv data/neuclir/qa-prompts-${split}-all.csv \
        --zero_shot --nextlines
done


