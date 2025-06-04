import time
import re
import unicodedata
import requests
import json
import argparse
from tqdm import tqdm

from neuclir_postprocess import *
from utils import citation_fixing
from utils import load_hits_tsv

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
    parser.add_argument("--candidates_tsv", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--submission", type=str, default=None)
    parser.add_argument("--quick_test", action='store_true', default=False)
    args = parser.parse_args()

    data_items = json.load(open(args.report_json, 'r'))
    if args.quick_test:
        data_items = data_items[:2]
    candidates = load_hits_tsv(args.candidates_tsv)

    outputs = []
    for item in tqdm(data_items, total=len(data_items)):
        # meta data
        request_id = str(item['requestid'])
        collection_ids = item['colectionids']
        lang_id = collection_ids.replace('neuclir/1/', '')[:2]
        cited_report = item['report']
        cited_report = citation_fixing(cited_report)
        references = [c['id'] for c in candidates[request_id + lang_id][:10]]

        # initialize an output placeholder
        output = ReportGenOutput(
            request_id=request_id,
            run_id=args.run_id,
            collection_ids=[collection_ids],
            raw_report=None,
            cited_report=cited_report,
            references=references
        )
        outputs.append(output)

    # prepare writer
    outputs = [o.finalize() for o in outputs]

    ## for checking 
    json.dump(outputs, open(args.submission+".json", 'w'), indent=4)

    ## for jsonl (official format)
    with open(args.submission+".jsonl", 'w', encoding='utf8') as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False)+ '\n')
