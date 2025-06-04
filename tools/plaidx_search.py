import time
import re
import unicodedata
import requests
import json
import argparse

def normalize_texts(texts):
    texts = unicodedata.normalize('NFKC', texts)
    texts = texts.strip()
    pattern = re.compile(r"\s+")
    texts = re.sub(pattern, ' ', texts).strip()
    pattern = re.compile(r"\n")
    texts = re.sub(pattern, ' ', texts).strip()
    return texts

def get_plaid_response(
    request_id,
    query,
    topk=100, 
    lang='zho', 
    writer=None
):
    url=f"""
        https://trec-neuclir-search.umiacs.umd.edu/query?query={query}&key=allInTheGroove33&content=true&limit={topk}&lang={lang}
    """
    response = requests.get(url.strip())
    response = json.loads(response.content)

    # query_ = response['query']
    system = response['system']
    results = response['results']

    hits = {'doc_ids': [], 'contents': [], 'scores': []}

    for i, result in enumerate(results):
        rank = i+1 

        doc_id = result['doc_id']
        normalized_content = normalize_texts(result['content'])

        hits['doc_ids'].append(doc_id)
        hits['contents'].append(normalized_content)

        if writer is not None:
            writer.write(f"{request_id}\t{lang_id[:2]}\t{rank}\t{doc_id}\t{normalized_content}\n")

    return hits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("-data", "--data_jsonl", type=str, default=None)
    parser.add_argument("-r", "--rewritten_json", type=str, default=None)
    parser.add_argument("-k", "--top_k", type=int, default=100)
    parser.add_argument("-out", "--output_tsv", type=str, default=None)
    args = parser.parse_args()

    # prepare query
    rewritten_queries = []
    if args.rewritten_json:
        data_items = json.load(open(args.rewritten_json))
        rewritten_queries = [item['rewrite'] for item in data_items]

    # prepare writer
    if args.output_tsv is not None:
        writer = open(args.output_tsv, 'w')
        writer.write(f"request_id\tlang\trank\tdoc_id\tsource_contents\n")
    else:
        writer = None

    # search
    hits_all = []

    with open(args.data_jsonl, 'r') as file:
        for i, line in enumerate(file):

            item = json.loads(line.strip())

            # query meta
            request_id = item['request_id']
            lang_id = item['collection_ids'][0].replace('neuclir/1/', '')

            # query text
            if args.rewritten_json is not None:
                query = rewritten_queries[i]
            else:
                query = item['problem_statement']

            start = time.time()
            item = json.loads(line.strip())
            # return collections
            hits = get_plaid_response(
                request_id=request_id,
                query=query,
                topk=args.top_k, 
                lang=lang_id,
                writer=writer
            )
            end = time.time()
            print(f"Search query - ({lang_id}) - {query[:100]} | Time elapsed: {(end-start):.2f}s")

            ### add a seperation because of google sheet limitation 
            if i % 30 == 0:
                writer.write('\n')

    if writer is not None:
        writer.close()

