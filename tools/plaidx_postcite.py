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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--report_json", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--submission", type=str, default=None)
    parser.add_argument("--quick_test", action='store_true', default=False)
    args = parser.parse_args()

    # prepare query
    data_items = json.load(open(args.report_json, 'r'))
    if args.quick_test:
        data_items = data_items[:2]

    outputs = []
    for item in tqdm(data_items, total=len(data_items)):
        # meta data
        request_id = str(item['requestid'])
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

        # one-shot verification
        # use statement (one-sentence) to search as provenances
        start = time.time()
        for idx_text, text in enumerate(output.texts):
            hits = get_plaid_response(
                request_id=request_id,
                query=text,
                topk=2, # at most 2
                lang=lang_id,
                writer=None
            )
            output.set_citations(idx_text, docids=hits['doc_ids'])

            if lang_id != 'fas':
                print('[sentence in report] -->', text)
                if len(hits['doc_ids']) > 0:
                    print('[provenance document] -->', hits['contents'][0][:200])
                else:
                    print('[provenance document] -->', 'NO search results')
                print('---')

        end = time.time()
        print(f"Search ({len(output.texts)}) sentences in the report - ({lang_id}) e.g. {output.texts[0][:30]}... | Time elapsed: {(end-start):.2f}s")

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

