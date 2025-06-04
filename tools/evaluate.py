from collections import defaultdict
import argparse
import json
import numpy as np
from utils import check_result

def main(args):

    output = json.load(open(args.result_data, 'r'))

    evaluation_all = defaultdict(list)
    N = len(output['data'])

    for data in output['data']:

        # print("===== PS: ", data['problem_statement'].strip())
        # print("===== BG:", data['background'].strip())
        evaluation = check_result(data['output'])
        evaluation_all['word_length'].append(evaluation['word_length'])


    print('# word length: ', np.mean(evaluation_all['word_length']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("-r", "--result_data", type=str, default=None)
    args = parser.parse_args()
    main(args)
