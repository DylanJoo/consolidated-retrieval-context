cd /home/dju/neuclir-irlab-ams/

# query rewriting prompts
for split in dev test;do
    python3 tools/prepare_gpt_rewrite.py \
        --input_jsonl data/requests-${split}-all.jsonl \
        --output_csv data/qr-prompts-${split}-all.csv
done

# question answering
for split in dev test;do
    python3 tools/prepare_gpt_answer.py \
        --input_jsonl data/requests-${split}-all.jsonl \
        --output_csv data/qa-prompts-${split}-all.csv \
        --zero_shot --nextlines
done


