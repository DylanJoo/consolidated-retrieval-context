import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import torch
import json
import argparse
import random
import re
from tqdm import tqdm
from collections import defaultdict

from ctxcomp.utils import load_model, batch_iterator
from tools.utils import load_hits_jsonl, normalize_texts

def rearrange(candidates, n_anchors=5, n_accompany=4):
    n_anchors = min(n_anchors, len(candidates))
    remaining = list(range(n_anchors, len(candidates)))

    new_candidates = []
    sampling = random.sample(remaining, len(remaining))

    for i in range(n_anchors):
        new_candidates.append(candidates[i])
        for j in sampling[:n_accompany]:
            new_candidates.append(candidates[j])
        sampling = sampling[n_accompany:]


    if len(sampling) > 0:
        new_candidates += sampling

    return new_candidates

def postprocess(mds, n=10):
    mds = normalize_texts(mds)

    l_bracket = re.findall(r'\[\d+\]', mds)
    for i in l_bracket:
        mds = mds.replace(i, '\n')

    r_bracket = re.findall(r'\[/\d+\]', mds)
    for i in r_bracket:
        mds = mds.replace(i, '\n')
    mds = mds.strip()

    pattern = re.compile(r"\n+")
    mds = re.sub(pattern, '\n', mds).strip()
    return mds

def truncate_and_concat(texts, tokenizer, max_length=512):
    tokenized = tokenizer.tokenize(texts)
    length = len(tokenizer.tokenize(texts))
    max_length = (max_length or tokenizer.max_lengt_single_sentence-1)
    if (length+6) < max_length:
        return texts
    else:
        return tokenizer.convert_tokens_to_string(tokenized[:(max_length-6)])

TEMPLATE = "Summarize context based on the topic. Write `unrelated` if context is irrelevant and write `redundant` if the information has been summarized in the former contexts. Topic: {} Context: [{}] {} [/{}]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_class", type=str, default=None, choices=["fid", "seq2seq", "causualLM"])
    parser.add_argument("--template", type=str, default=TEMPLATE)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--candidate_jsonl", type=str, default=None)
    parser.add_argument("--output_key", type=str, default="recomp_summary")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # load model
    model, tokenizer = load_model(args.model_name_or_path, model_class=args.model_class)
    model.eval()

    # load writer and evaluation data 
    eval_data = [json.loads(line.strip()) for line in open(args.eval_file).readlines()]
    candidates = load_hits_jsonl(args.candidate_jsonl)
    writer = open(args.output_file, 'w')

    for eval_item in tqdm(eval_data, total=len(eval_data)):

        # multiple retrieved document here
        request_id = eval_item['request_id']
        lang = eval_item['collection_ids'][0].replace('neuclir/1/', '')[:2]
        request = eval_item['problem_statement']
        candidate_items = [d for d in candidates[request_id + lang]]
        candidate_items = rearrange(
            candidates=candidate_items, n_anchors=10, n_accompany=2
        )
        docs = [c['translation'] for c in candidate_items]
        docs = [truncate_and_concat(doc, tokenizer, max_length=900) for doc in docs]
        ids = [c['id'] for c in candidate_items]
        qids = [c['qid'] for c in candidate_items]

        predictions = []
        logger.info(f'Rearraged indices and documents: {len(ids)} and {len(docs)}')
        for batch_docs in batch_iterator(docs, 3):

            input = []
            for i, doc in enumerate(batch_docs):
                input += [args.template.format(request, i+1, doc, i+1)]

            tokenized_input = tokenizer(
                input, 
                max_length=args.max_length-1, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(model.device)

            tokenized_input['input_ids'] = tokenized_input['input_ids'].view(
                -1, len(batch_docs), tokenized_input['input_ids'].size(-1)
            )
            tokenized_input['attention_mask'] = tokenized_input['attention_mask'].view(
                -1, len(batch_docs) * tokenized_input['attention_mask'].size(-1)
            )

            output = model.generate(
                **tokenized_input,
                top_p=0.1,
                temperature=0.2,
                min_new_tokens=64, 
                max_new_tokens=args.max_length
            )
            prediction = tokenizer.decode(output[0], skip_special_tokens=True) 

            # postprocess
            # postprocessed = prediction
            postprocessed = [p.strip() for p in postprocess(prediction, len(input)).split('\n') if len(p) > 3]
            logger.info(prediction)
            predictions += postprocessed

            logger.info(f"{len(postprocessed)} {len(batch_docs)}")
            offset = len(postprocessed) - len(batch_docs)
            if offset < 0:
                predictions += [" "] * (-offset)

            if offset > 0:
                predictions = predictions[:-offset]

        ## finalize
        for cand, text in zip(candidate_items, predictions):
            cand.update({args.output_key: text})
            writer.write(json.dumps(cand, ensure_ascii=False) + '\n')

    writer.close()

if __name__ == '__main__':
    with torch.no_grad():
        main()

