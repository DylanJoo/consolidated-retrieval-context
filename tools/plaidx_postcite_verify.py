import time
import re
import unicodedata
import requests
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

from neuclir_postprocess import *
from plaidx_search import get_plaid_response

def normalize_texts(texts):
    texts = unicodedata.normalize('NFKC', texts)
    texts = texts.strip()
    pattern = re.compile(r"\s+")
    texts = re.sub(pattern, ' ', texts).strip()
    pattern = re.compile(r"\n")
    texts = re.sub(pattern, ' ', texts).strip()
    return texts

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def init_verifier():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return tokenizer, model

def verify_statement(
    model, 
    tokenizer, 
    statement, 
    evidences, 
    batch_size=64, 
    max_length=512
):
    scores = []
    for batch_premise in batch_iterator(evidences, batch_size):
        batch_hypothesis = [statement] * len(batch_premise)
        print(batch_hypothesis[0], "-->", batch_premise[0])
        input = tokenizer(
            batch_premise, batch_hypothesis, 
            truncation="only_first", # keep the second one intact
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            predictions = model(**input).logits.softmax(-1)
            label_names = ["entailment", "neutral", "contradiction"]
            entail_scores = predictions[:, 0].tolist()
            scores += entail_scores

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--report_json", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--submission", type=str, default=None)
    parser.add_argument("--quick_test", action='store_true', default=False)
    # verification 
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_word_length", type=int, default=50)
    parser.add_argument("--fact_threshold", type=float, default=0.7)
    args = parser.parse_args()

    # prepare query
    data_items = json.load(open(args.report_json, 'r'))
    if args.quick_test:
        data_items = data_items[:2]

    outputs = []
    for item in tqdm(data_items, total=len(data_items)):
        # meta data
        request_id = item['requestid']
        collection_ids = item['collectionids']
        lang_id = collection_ids.replace('neuclir/1/', '')
        raw_report = item['report']

        # initialize an output placeholder
        output = ReportGenOutput(
            request_id=request_id,
            run_id=f"{lang_id}_{args.run_id}",
            collection_ids=[collection_ids],
            raw_report=raw_report,
            cited_report=None
        )

        # two-step verification
        ## extract snippets (multiple sentences) to search as provenance candidates
        start = time.time()
        snippets = output.get_snippets(max_word_length=args.max_word_length)
        for i, snippet in enumerate(snippets):
            hits = get_plaid_response(
                request_id=request_id,
                query=snippet,
                topk=args.top_k,
                lang=lang_id,
                writer=None
            )
            ### append new documents
            if i == 0:
                hits_pool = copy(hits)
            else:
                for doc_id, content in zip(hits['doc_ids'], hits['contents']):
                    if doc_id not in hits_pool['doc_ids']:
                        hits_pool['doc_ids'] += [doc_id]
                        hits_pool['contents'] += [content]

        output.set_references(hits_pool['doc_ids'])  # reference index start with 1
        end = time.time()
        print(f"Search ({i}) snippets using ({len(output.texts)}) sentences, ({len(hits_pool['doc_ids'])}) provenance candidates collected")
        print(f"Searching time elapsed: {(end-start):.2f}s")

        ## rerank/verify candidates as citations
        tokenizer, model = init_verifier()
        for idx_text, text in enumerate(output.texts):
            scores = verify_statement(
                model=model, tokenizer=tokenizer,
                statement=text, 
                evidences=hits_pool['contents'],
                batch_size=args.batch_size, 
                max_length=args.max_length
            )
            reference_ids = [str(i+1) for i in np.argsort(scores)[::-1][:2] \
                    if scores[i] >= args.fact_threshold ]
            output.set_citations(idx_text, referenceids=reference_ids)

            if lang_id != 'fas':
                reference_indices = [i for i in np.argsort(scores)[::-1][:2] \
                        if scores[i] >= args.fact_threshold ]

                print('[sentence in report] -->', text)
                if len(reference_indices) > 0:
                    idx = reference_indices[0]
                    print(f'[provenance document (entail: {scores[idx]:.2f}) -->', \
                            hits_pool['contents'][idx][:100])
                else:
                    print('[provenance document] -->', 'NO qualified results')
                print('---')

        end_v = time.time()
        print(f"Verification time elapsed: {(end_v-end):.2f}s")

        # append the output of a topic
        outputs.append(output)

    # prepare writer
    outputs = [o.finalize() for o in outputs]
    ## for jsonl (official format)
    with open(args.submission+".jsonl", 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')

    ## for checking 
    json.dump(outputs, open(args.submission+".json", 'w'), indent=4)

