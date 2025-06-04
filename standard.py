import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import yaml
import argparse
import json
import numpy as np
from tqdm import tqdm

from llm.base import LLM
from prompts.neuclir import *
from tools.utils import load_hits_tsv, load_hits_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")

    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--train_file", type=str, help="Path to the eval file")
    parser.add_argument("--eval_file", type=str, help="Path to the eval file")
    parser.add_argument("--candidates_tsv", type=str, help="Path to the eval file")
    parser.add_argument("--candidates_jsonl", type=str, help="Path to the eval file")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--split", type=str, default='dev')

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents, the exact number will go in decoder.")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--closebook", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")

    # Model and name
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--load_mode", type=str, default='no', help="Model to use")
    parser.add_argument("--attn_implementation", type=str, default='flash_attention_2')

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use original/translation of the documents
    parser.add_argument("--used_field", type=str, default="target_contents", help="Use compressed text data. Option: `target_contents`, `translation`")

    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if "turbo" in args.model:
        args.max_length = 4096
    if "16k" in args.model:
        args.max_length = 16384
    elif "32k" in args.model:
        args.max_length = 32768
    elif "turbo" in args.model:
        args.max_length = 4096
    elif "gpt-4" in args.model:
        args.max_length = 8192
    elif "llama-2" in args.model.lower() or "llama2" in args.model.lower():
        args.max_length = 4096
    elif "llama-3" in args.model.lower() or "llama3" in args.model.lower():
        args.max_length = 8192
    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")

    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    # Load the model or setup the API
    llm = LLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load data
    # train_data = json.load(open(args.train_file))
    eval_data = [json.loads(line.strip()) for line in open(args.eval_file).readlines()]
    if args.candidates_tsv:
        candidates = load_hits_tsv(args.candidates_tsv)
    if args.candidates_jsonl:
        candidates = load_hits_jsonl(args.candidates_jsonl, key=args.used_field)

    ## Prepare instruction and the demo prompts
    # Sample quick test
    if args.quick_test is not None:
        np.random.seed(args.seed)
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]

    logger.info("Generating prompts...") 
    for idx, eval_item in enumerate(tqdm(eval_data)):

        request_id = eval_item['request_id']
        lang = eval_item['collection_ids'][0].replace('neuclir/1/', '')[:2]

        ## preprocess
        demo_prompt = ""

        ### ZS-closebook
        if args.closebook:
            promt = apply_closebook_prompt(
                PS=eval_item["problem_statement"], 
                BG=eval_item["background"],
                LIMIT=eval_item["limit"],
                R="", INST=rag_instruction
            )
        ### ZS-rag and 1S-rag (may overlength)
        else:
            doc_prompt = apply_docs_prompt(
                doc_items=candidates[request_id + lang],
                ndoc=args.ndoc,
                field=args.used_field,
                max_length=args.max_doc_length
            )
            prompt = apply_rag_prompt(
                PS=eval_item["problem_statement"], 
                BG=eval_item["background"],
                LIMIT=eval_item["limit"],
                D=doc_prompt,
                R="", INST=rag_instruction
            )
        prompt = prompt.replace("{DEMO}", demo_prompt)
        eval_data[idx]['prompt'] = prompt
        eval_data[idx]['lang'] = lang
        doc_ids = [c["id"] for c in candidates[request_id + lang]]
        eval_data[idx]['references'] = doc_ids

    logger.info("Done prompt preparation.")
    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']
        prompt_len = len(llm.tokenizer.tokenize(prompt))

        if idx == 0:
            print(prompt)

        output_array = []
        for _ in range(args.num_samples):
            output_array.append(llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
            item['prompt'] = prompt
            item['prompt_len'] = prompt_len
            
            output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
            if output_array[-1].endswith("End."):
                output_array[-1] = output_array[-1][:-len("End.")]

            logger.info(f"Question: {item['problem_statement'][:40]}")
            logger.info(f"Prompt text (length={prompt_len}): {prompt}")
            logger.info(f"Final model output: {output_array[-1]}") 
        
        item['output'] = output_array if len(output_array) > 1 else output_array[0]
        
    logger.info(f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}")
    logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")

    # Save the result
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"report-{args.tag}-{args.split}-all"

    eval_data = {"args": args.__dict__, "data": eval_data}

    if not os.path.exists("results"):
        os.makedirs("results")

    json.dump(eval_data, open("results/" + name + ".json", "w"), indent=4)

if __name__ == "__main__":
    main()

