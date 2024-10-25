# read all lines from all files, merge them, shuffle the, and write to a single file


import glob
import json
from tqdm import tqdm
import os
all_files = glob.glob('/work/nvme/bbvf/mtian8/LLM/Megatron-LM/processed/*/train.jsonl')

print('all files:', all_files)

all_samples = []
for f in tqdm(all_files, desc="Processing files"):
    with open(f) as fp:
        lines = fp.readlines()
        for line in tqdm(lines, desc=f"Processing lines in {f}"):
            sample = json.loads(line)
            all_samples.append(sample)

print('all samples:', len(all_samples))

# shuffle
import random
random.shuffle(all_samples)

# write to a single file
with open('/work/nvme/bbvf/mtian8/LLM/Megatron-LM/processed/merged_shuffle_train.jsonl', 'w') as fp:
    for sample in tqdm(all_samples, desc="Writing to Output"):
        fp.write(json.dumps(sample))
        fp.write('\n')