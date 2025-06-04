from tqdm import tqdm
import argparse
import json

template = """
Formulate a search query for the given user request. The user's background is also provided (some information might help customize a more engaging search query). The query will serve as keywords and pass through a news passage searching engine. The query should be comprehensive and clearly indicate the information needs. Please ensure that the query is formulated in 100 words or fewer.\n\n### User request: {REQUEST}\n\n### User background: {BACKGROUND}\n\n### Query:""".strip()

template = template.replace('\n', '\\n')

def prepare(args):

    writer = open(args.output_csv, 'w') 
    writer.write("request_id\tcolection_ids\tprompt\n")

    eval_data = []
    with open(args.input_jsonl, 'r') as file:
        for line in tqdm(file):
            item = json.loads(line.strip())
            eval_data.append(item)
            # request_id = item['requrest_id']
            # collection_ids = item['collection_ids']
            # background = item['background'] 
            # problem = item['problem_statement'] 
            # limit = item['limit']

    for i, eval_item in enumerate(eval_data):
        id = eval_item['request_id']
        prompt = template.replace("{REQUEST}", eval_item['problem_statement'])
        prompt = prompt.replace("{BACKGROUND}", eval_item['background'])
        lang = eval_item['collection_ids']

        if len(lang) > 1:
            print('more than one collection ids')
            exit(0)

        writer.write(f"{id}\t{lang[0]}\t{prompt}\n")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("-input", "--input_jsonl", type=str, default=None)
    parser.add_argument("-output", "--output_csv", type=str, default=None)
    args = parser.parse_args()

    prepare(args)
