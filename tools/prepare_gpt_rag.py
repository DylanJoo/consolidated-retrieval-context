from tqdm import tqdm
import argparse
import json
from prompts.neuclir import *
from utils import load_hits_tsv

def prepare(args):

    writer = open(args.output_csv, 'w') 
    writer.write("request_id\tcollection_ids\tprompt\n")

    eval_data = [json.loads(line.strip()) for line in open(args.input_jsonl).readlines()]
    candidates = load_hits_tsv(args.candidates_tsv)  

    for i, eval_item in enumerate(eval_data):
        request_id = eval_item['request_id']
        collection_id = eval_item['collection_ids'][0]
        lang = collection_id.replace('neuclir/1/', '')[:2]

        doc_prompt = apply_docs_prompt(
            doc_items=candidates[request_id + lang],
            ndoc=args.ndoc,
            field='translation',
            max_length=args.max_doc_length
        )
        prompt = apply_rag_prompt(
            PS=eval_item['problem_statement'],
            BG=eval_item['background'],
            LIMIT=eval_item['limit'],
            D=doc_prompt,
            R="", INST=rag_instruction
        )
        prompt = prompt.replace("{DEMO}", "")

        if args.nextlines:
            prompt = display_nextlines(prompt)

        writer.write(f"{request_id}\t{collection_id}\t{prompt}\n")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--candidates_tsv", type=str, default=None)
    parser.add_argument("--ndoc", type=int, default=10)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--nextlines", action='store_true', default=False)
    args = parser.parse_args()

    prepare(args)
