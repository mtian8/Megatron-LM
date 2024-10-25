#!/usr/bin/env python3
""" load all stat.json files and print the content in a table"""


import glob
import json


def main():
    files = glob.glob('processed/*/stat.json')
    # example result
    """
    {
    "num_samples": 7969,
    "num_concat_samples": 3900,
    "sum_system_prompt_tokens": 231101,
    "sum_question_tokens": 3586753,
    "sum_quesiion_count": 7969,
    "avg_question_length": 450.088216840256,
    "sum_reply_tokens": 4251622,
    "sum_reply_count": 7969,
    "avg_reply_length": 533.5201405446104,
    "avg_reply_count_per_conversation": 1.0,
    "avg_question_count_per_conversation": 1.0,
    "input_files": [
        "raw/rosetta_alpaca.json"
    ],
    "output_dir": "processed/rosetta_alpaca",
    "total_tokens": 7987200,
    "total_loss_tokens": 4257528
    }
    
    """

    columns = [
        'file',
        'num_samples',
        'total_tokens',
        'total_loss_tokens',
        'avg_question_length',
        'avg_question_count_per_conversation',
        'avg_reply_length',
        'avg_reply_count_per_conversation',
        'input_files',
    ]

    all_data = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            all_data.append(data)

    # csv format
    print(','.join(columns))
    for data in all_data:
        print(data['output_dir'], end=',')
        print(','.join([str(data[k]) for k in columns[1:]]))
            

if __name__ == '__main__':
    main()