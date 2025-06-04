from tqdm import tqdm
import argparse
import json

instruction_prompt = """
Write an engaging report for the given request within {LIMIT} words. The request contains a problem statement and the requester background (some information might help customize the report). Use an unbiased and journalistic tone. Always write the report based on facts.
""".strip()

inst_prompt_template = "{INST}\n\n\n{DEMO}Request:\nProblem statement: {PS}\nRequester background: {BG}\n\nReport: {R}"

def display_nextlines(texts):
    texts = texts.replace('\n', '\\n')
    return texts

def apply_inst_prompt(PS, BG, LIMIT=100, R="", INST=""):
    p = inst_prompt_template
    p = p.replace("{INST}", INST).replace("{LIMIT}", str(LIMIT//10)).strip()
    p = p.replace("{PS}", PS).replace("{BG}", BG).replace("{R}", R)
    return p

def prepare(args):

    writer = open(args.output_csv, 'w') 
    writer.write("request_id\tcolection_ids\tprompt\n")

    eval_data = []
    with open(args.input_jsonl, 'r') as file:
        for line in tqdm(file):
            item = json.loads(line.strip())
            eval_data.append(item)

    for i, eval_item in enumerate(eval_data):
        id = eval_item['request_id']
        lang = eval_item['collection_ids']

        if args.zero_shot:
            prompt = apply_inst_prompt(
                PS=eval_item['problem_statement'],
                BG=eval_item['background'],
                LIMIT=eval_item['limit'],
                R="", INST=instruction_prompt
            )
            prompt = prompt.replace("{DEMO}", "")

        # sanity check
        if len(lang) > 1:
            print('more than one collection ids')
            exit(0)

        if args.nextlines:
            prompt = display_nextlines(prompt)

        writer.write(f"{id}\t{lang[0]}\t{prompt}\n")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("-input", "--input_jsonl", type=str, default=None)
    parser.add_argument("-output", "--output_csv", type=str, default=None)
    # closebook QA
    parser.add_argument("-zs", "--zero_shot", action='store_true', default=False)
    parser.add_argument("-nl", "--nextlines", action='store_true', default=False)
    args = parser.parse_args()

    prepare(args)
