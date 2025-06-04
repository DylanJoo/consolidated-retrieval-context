import argparse
import json
from neuclir_postprocess import ReportGenOutput

def load_requests():
    dev_requests = {}
    with open('data/requests-dev-all.jsonl') as f:
        for line in f:
            dev_item = json.loads(line)
            dev_requests[dev_item['request_id']] = {
                "background": dev_item['background'],
                "problem_statement": dev_item['problem_statement']
            }
    return dev_requests

def convert_train_data(requests=None):
    data_items = [json.loads(d) for d in open('data/NeuCLIR-pilot2024.json').readlines()] 
    outputs = []
    for item in data_items:
        output = ReportGenOutput(
            request_id=item['request_id'],
            run_id='NeuCLIR-pilot2024',
            collection_ids=item['collection_ids'],
            texts=[sent[0] for sent in item['report']],
            citations=[sent[1] for sent in item['report']],
        )
        ## add references 
        references = list(set( sum([sent[1][:2] for sent in item['report']], []) ))
        output.set_references(docids=references, shuffle=True)
        outputs.append(output)
    return outputs

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--convert_pilot_data", default=False, action='store_true')
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    ## create the prompt 
    if args.convert_pilot_data:
        train_data = convert_train_data()
        requests = load_requests()
        outputs = []
        for data in train_data:
            references = data.get_references()
            try:
                request = requests[data.request_id]
            except:
                print(f'Ignore this request {data.request_id}, as no backgroun or problem found')
                continue

            report = ""
            for i, text in enumerate(data.texts):
                report += text

                if text.strip().endswith('.'):
                    report = report[:-1]
                    dot = True

                reference_text = ""
                for referenceid in references[i]:
                    reference_text += f"[{referenceid}]"

                report += f" {reference_text}"
                if dot:
                    report += f". "
                    dot = False

            train_data_dict = {
                'request_id': data.request_id,
                'collection_ids': data.collection_ids,
                'problem_statement': request['problem_statement'],
                'background': request['background'],
                'report': report, 
                'references': data.references
            }
            outputs.append(train_data_dict)

    elif 'jsonl' in args.input:
        outputs = load_jsonl(args.input)
    else:
        outputs = json.load(open(args.input, 'r'))

    if 'jsonl' in args.output:
        with open(args.output, 'w') as f:
            for output in outputs:
                f.write(json.dumps(output.finalize()) + '\n')
    else:
        with open(args.output, 'w') as f:
            json.dump(outputs, f, indent=4)
