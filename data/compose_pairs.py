import os
import re
import argparse
import json
import numpy as np
from copy import copy
import random
from collections import defaultdict
from tqdm import tqdm
from glob import glob
from pyserini.search.lucene import LuceneSearcher

from utils import (
    normalize_texts, 
    maybe_truncation,
    load_topic,
    load_passages,
)

def deduplicate_and_sort(doc_ids):
    doc_ids = list(set(doc_ids))
    sorted(doc_ids)
    return doc_ids

def check_newinfo(values, new_values):
    mask = (values == 0)
    if (new_values[mask] > values[mask]).any():
        return True, values + new_values
    else:
        return False, values

def get_i_doc(i, psgs_bound):
    for i_doc, bound in enumerate(psgs_bound):
        if i in bound:
            return i_doc

def binary_rerank_greedy(item_, threshold=3):
    item = copy(item_)
    example_id = item['id']
    ratings = np.array(item['ratings'])
    passages = item['passages']
    labels = {"documents": [], "passages": [], "redundant_passages": []}

    # binariz
    scores = np.zeros_like(ratings).astype(int)
    scores[(ratings >= threshold)] = 1
    answerable = (scores.sum(0) != 0)

    ## select the best passages per doc
    end = np.cumsum([len(psgs) for psgs in passages])
    start = np.append([0], end)[:(-1)]
    psgs_in_doc = [range(s, e) for (s, e) in zip(start, end)]

    ## initial values
    values = np.zeros(scores.shape[1])
    while (values[answerable]==0).sum() > 0:

        ## rerank the passages with binary + max
        mask = (values == 0)
        bin_counts = np.where(ratings[:, mask]>=threshold, 1, 0).sum(-1) 
        max_scores = np.where(ratings[:, mask]>0, 1, 0).max(-1)
        psg_scores = bin_counts * 1 + max_scores * 0.1
        i = psg_scores.argmax()

        flag, values = check_newinfo(values, scores[i])
        i_doc = get_i_doc(i, psgs_in_doc)

        if flag:
            if i_doc not in labels["documents"]:
                labels["documents"].append(i_doc)
            labels["passages"].append( (i_doc, i) )

            # strictly duplicated
            for j, r in enumerate(ratings):
                if (ratings[i] > r).all():
                    labels['redundant_passages'].append( (i_doc, j) )

    return labels

def minmax_rerank_greedy(scores):
    pass
def reciprocal_rerank_greedy(scores):
    pass

def mine_distractor(
    query, 
    searcher, 
    k=10,
    max_docs_return=3,
    ignored_prefix=""
):

    hits = searcher.search(query, k)

    hit_doc_ids = []
    ### schema: {example_id}#{n_doc}:{n_claims}
    for h in hits:
        if h.docid.split("#")[0] != ignored_prefix:
            hit_doc_id = h.docid.split(":")[0]
            if hit_doc_id not in hit_doc_ids:
                hit_doc_ids.append(hit_doc_id)
            if len(hit_doc_ids) >= max_docs_return:
                return hit_doc_ids

    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--dataset_jsonl", type=str, default=None)
    parser.add_argument("--alignment_jsonl", type=str, default=None)
    parser.add_argument("--n_max_distractors", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--doc_passages_index", type=str)
    parser.add_argument("--good_read", default=False, action='store_true')

    args = parser.parse_args()
    os.makedirs(args.dataset_jsonl.rsplit('/', 1)[0], exist_ok=True) 
    writer = open(args.dataset_jsonl, 'w')

    # topic (request is a special case for report gen)
    topic_all = []
    files = glob(os.path.join(args.input_dir, "*request*.json"))
    for file in tqdm(files):
        topic = load_topic(file)
        topic_all += topic
    topic_all = {r['example_id']: r['texts'] for r in topic_all}

    # documents 
    document_all = []
    files = glob(os.path.join(args.input_dir, "*summ*.json"))
    for file in tqdm(files):
        item = load_passages(file)
        document_all += item
    document_all = {r['example_id']: r['docs_full_texts'] for r in document_all}

    # alignment 
    alignment_all = []
    with open(args.alignment_jsonl) as f:
        for line in f :
            item = json.loads(line.strip())
            ratings = np.array(item['ratings'])
            alignment_all.append({
                "id": item['example_id'],
                "topic": topic_all[item['example_id']],
                "documents": document_all[item['example_id']],
                "passages": item['passages'],
                "dis_documents": [],
                "questions": item['questions'],
                "ratings": ratings,
            })
            shape = (sum([len(psgs) for psgs in item['passages']]), len(item['questions']))
            assert shape == ratings.shape, 'inconsistent shape'

    # mine distractor
    if args.n_max_distractors > 0:
        searcher = LuceneSearcher(args.doc_passages_index)
        for alignment in alignment_all:
            dis_example_doc_ids = mine_distractor(
                query=alignment['topic'],
                searcher=searcher, 
                k=10,
                max_docs_return=args.n_max_distractors,
                ignored_prefix=alignment['id']
            )
            for example_doc_id in dis_example_doc_ids:
                example_id, num = example_doc_id.split('#')
                alignment["dis_documents"].append(
                    document_all[example_id][int(num)]
                )
                # alignment["ratings"] = ?????


    ## compose into pairs
    for alignment in tqdm(alignment_all):

        example_id = alignment['id']
        ndocs = len(alignment['documents'])
        documents = alignment['documents']
        passages = [p for psgs in alignment['passages'] for p in psgs]

        labels = binary_rerank_greedy(alignment, threshold=3)

        docids = [f"{example_id}#{i}" for i in labels['documents']]
        psgids = [f"{example_id}#{i}:{j}" for (i, j) in labels['passages']]

        full_ctxs, full_ctxs_r = [], []
        for i in range(ndocs):
            if i in labels['documents']:
                full_ctxs.append(documents[i])
            else:
                full_ctxs_r.append(documents[i])

        comp_ctxs = [[] for _ in labels['documents']]
        for i, i_doc in enumerate(labels['documents']):
            for (ii, j) in labels['passages']:
                if ii == i_doc:
                    comp_ctxs[i].append(passages[j])

        full_ctxs_d = alignment['dis_documents'][:args.n_max_distractors]

        writer.write(json.dumps({
            "topic": alignment['topic'],
            "docids": docids,
            "psgids": psgids,
            "doc_ctxs": full_ctxs,
            "comp_ctxs": comp_ctxs,
            "distract_ctxs": full_ctxs_d,
            "redundant_ctxs": full_ctxs_r,
        }, indent=4 if args.good_read else None)+'\n')

    writer.close()

    print('done')
