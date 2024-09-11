import os
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union
from glob import glob
import datasets as hf_datasets
import psutil
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from functools import partial
import os
import warnings
from typing import Dict, Iterable, Union
import datasets
import datasets as hf_datasets
import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase
import json

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ConcatTokensDataset(IterableDataset):
    def __init__(
            self,
            dataset_iter,
            max_length: int,
            no_wrap: bool = False,
    ):
        self.dataset_iter = dataset_iter
        self.should_wrap = not no_wrap
        self.max_length = max_length

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        input_ids_buffer = []
        loss_mask_buffer = []
        for sample in self.dataset_iter:
            # process token_ids and loss_mask
            input_ids = sample["token_ids"]
            loss_mask = sample["loss_mask"]
            input_ids_buffer = input_ids_buffer + input_ids
            loss_mask_buffer = loss_mask_buffer + loss_mask
            while len(input_ids_buffer) >= self.max_length:
                concat_input_ids = input_ids_buffer[:self.max_length]
                concat_loss_mask = loss_mask_buffer[:self.max_length]
                input_ids_buffer = input_ids_buffer[self.max_length:] if self.should_wrap else []
                loss_mask_buffer = loss_mask_buffer[self.max_length:] if self.should_wrap else []
                yield {
                    'token_ids': concat_input_ids,
                    'loss_mask': concat_loss_mask
                }

DEFAULT_SYSTEM_PROMPT = """You are a helpful and intelligent assistant. Follow the instructions and answer the questions as accurately as possible."""
CONCAT_LENGTH = 2048

from transformers import AutoTokenizer


def split_into_paragraphs(text, min_length=200000):
    paragraphs = []
    current_paragraph = ""

    i = 0
    while i < len(text):
        current_paragraph += text[i]

        # Check if the end of paragraph is reached
        if i + 1 < len(text) and text[i] == '\n' and text[i + 1] == '\n':
            if len(current_paragraph) >= min_length:
                paragraphs.append(current_paragraph)
                current_paragraph = ""
            i += 1  # Skip the next newline as it's part of the paragraph break

        i += 1

    # Add the last paragraph if it's not empty
    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs


def process_func_base(name, input_files, output_dir, tokenizer_path):
    # print(f'Processing: {name}')
    logger.info('Processing: %s', name)

    stats = {
        'num_samples': 0,
        'num_concat_samples': 0,
        'sum_system_prompt_tokens': 0,
        'sum_question_tokens': 0,
        'sum_quesiion_count': 0,
        'avg_question_length': None,
        'sum_reply_tokens': 0,
        'sum_reply_count': 0,
        'avg_reply_length': None,
        'avg_reply_count_per_conversation': None,
        'avg_question_count_per_conversation': None,
        'input_files': input_files,
        'output_dir': output_dir,
    }

    ###################### tokenizer setup start
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              trust_remote_code=True)
    
    # the tokenizer should not add bos and eos tokens, this should be set in the config already
    # just to make sure, we disable the bos and eos tokens again
    
    assert tokenizer.add_bos_token == False
    assert tokenizer.add_eos_token == False

    bos_token = tokenizer.bos_token
    bos_token_id = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    sys_start = '<|sys_start|>'
    sys_start_id = tokenizer.convert_tokens_to_ids(sys_start)
    sys_end = '<|sys_end|>'
    sys_end_id = tokenizer.convert_tokens_to_ids(sys_end)
    im_start = '<|im_start|>'
    im_start_id = tokenizer.convert_tokens_to_ids(im_start)
    im_end = '<|im_end|>'
    im_end_id = tokenizer.convert_tokens_to_ids(im_end)

    # # tokenizer has the following special tokens:
    # assert bos_token_id == 1
    # assert eos_token_id == 2
    # assert im_start_id == 32018
    # assert im_end_id == 32019
    # assert sys_start_id == 32020
    # assert sys_end_id == 32021

    ###################### tokenizer setup end

    all_documents = []  # untokenized documents, each sample is a json object

    """
    each document will be tokenized and form a sample, the length will vary at this stage.
    each sample will be like:
    {
        "token_ids": [1, 1000, 1001, 1002, 1003, 1004, 1005, ...],  # 1 is the bos token
        "label_ids": [1001, 1002, 1003, 1004, 1005, 1006, ..., 2],  # 2 is the eos token
        "loss_mask": [0, 0, 0, 0, 0, 0, ..., 1, 1, 1, 1, 1, 1, 1, 1, 1],    # 0 for tokens to be masked, 1 for tokens to be predicted
    }
    """
    # MAX_SAMPLE_COUNT = 100000000000000000000 # set a large number to process all samples, debug purpose
    all_tokenized_documents = []
    for input_file in input_files:
        logger.info('Processing file: %s', input_file)
        if input_file.endswith('.jsonl'):
            with open(input_file, 'r') as fin:
                for line in fin:
                    all_documents.append(json.loads(line))
                #     if len(all_documents) >= MAX_SAMPLE_COUNT:
                #         break
                # if len(all_documents) >= MAX_SAMPLE_COUNT:
                #     break
        elif input_file.endswith('.json'):
            with open(input_file, 'r') as fin:
                all_documents.extend(json.load(fin))
        else:
            # a directory
            all_documents = datasets.load_dataset(input_file)['train']

    # print('Number of documents: ', len(all_documents))
    logger.info('Number of documents: %s', len(all_documents))

    def debug_print(x):
        print('>>> token_ids decode:\n', tokenizer.decode(x['token_ids']))
        # print('>>> label_ids decode:\n', tokenizer.decode(label_ids))
        # print('loss_mask: ', loss_mask)
        learn_labels = []

        for label_id, mask in zip(x['token_ids'], x['loss_mask']):
            if mask == 1:
                learn_labels.append(label_id)
            else:
                learn_labels.append(tokenizer.convert_tokens_to_ids('_'))
        print('>>> learn_labels:\n', tokenizer.decode(learn_labels))

    def process_sample1(sample):
        # this is implemented to follow the SlimOrca format, which is instruct format

        conversations = sample['conversations']

        """
        sample example:
        conversations = [
            {'from': 'system', 'value': 'You are an AI assistant. You will be given a task. You must generate a detailed and long answer.'}
            {'from': 'human', 'value': 'What is the best order to watch the Star Wars series? ...', ...}
            {'from': 'gpt', 'value': 'The best', ...}
        ]
        """

        """
        Target format:
        list special tokens:
        bos: <s> 
        eos: </s>
        system_start: <|sys_start|>
        system_end: <|sys_end|>
        user_start: <|im_start|>
        user_end: <|im_end|>

        Note: 
        Pay attention to the spaces, ideally no spaces before or after special tokens, but to better visualize, we add spaces here.
        The special tokens are added using the token ids (for example, will directly add 1 for bos, instead of adding "<s>" to the text).

        Format:
        <s> <|sys_start|> system prompt <|sys_end|> <|im_start|> first user utterance <|im_end|> first model response <|im_start|> next user utterance <|im_end|> next model response </s>
        """

        sample_tokenized = []
        sample_loss_mask = []  # at the stage, the loss mask is aligned with the tokenized text. Will shift left by one after all tokens are concatenated

        # add bos token
        sample_tokenized.append(bos_token_id)
        sample_loss_mask.append(0)

        # determine the system prompt
        if conversations[0]['from'] == 'system':
            system_prompt = conversations[0]['value']
            conversations = conversations[1:]  # remove the first system prompt
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        # tokenize system prompt
        system_prompt_tokenized = [sys_start_id] + tokenizer(system_prompt, truncation=False, padding=False)[
            'input_ids'] + [sys_end_id]  # <|sys_start|> system prompt <|sys_end|>
        system_prompt_loss_mask = [0] * len(system_prompt_tokenized)  # no loss for system prompt

        sample_tokenized += system_prompt_tokenized
        sample_loss_mask += system_prompt_loss_mask

        # stats for sys prompt
        stats['sum_system_prompt_tokens'] += len(system_prompt_tokenized)

        for message in conversations:
            if message['from'] == 'human':
                # tokenize user uttergit@github.com:mtian8/Megatron-LM.gitance
                user_utterance_tokenized = [im_start_id] + tokenizer(message['value'], truncation=False, padding=False)[
                    'input_ids'] + [im_end_id]  # <|im_start|> first user utterance <|im_end|>
                user_utterance_loss_mask = [0] * len(user_utterance_tokenized)  # no loss for user utterance
                sample_tokenized += user_utterance_tokenized
                sample_loss_mask += user_utterance_loss_mask

                # stats for user utterance
                stats['sum_question_tokens'] += len(user_utterance_tokenized)
                stats['sum_quesiion_count'] += 1

            elif message['from'] == 'gpt':
                # tokenize model response
                # Note: There is no eos at the end of model response. The eos should mark the end of the whole sample, not the end of model response.
                # Note: To indicate the end of model response, the model should learn to generate the <|im_start|> token.
                model_response_tokenized = tokenizer(message['value'], truncation=False, padding=False)['input_ids']
                model_response_loss_mask = [1] * len(
                    model_response_tokenized)  # should predict all tokens in model response
                sample_tokenized += model_response_tokenized
                sample_loss_mask += model_response_loss_mask

                # stats for model response
                stats['sum_reply_tokens'] += len(model_response_tokenized)
                stats['sum_reply_count'] += 1

        sample_tokenized.append(eos_token_id)
        sample_loss_mask.append(1)  # should predict the eos token

        assert len(sample_tokenized) == len(sample_loss_mask)

        result = {
            "token_ids": sample_tokenized,
            "loss_mask": sample_loss_mask
        }

        debug = False
        if debug:
            debug_print(result)

        return result

    def process_sample2(sample):
        # this is implemented to follow the program_books format, which is just plain text
        # use sample['markdown']

        # sometimes the text is just too long, the tokenizer will be super slow.
        # we first split the text into paragraphs, and tokenize each paragraph separately
        text = sample['markdown']

        if len(text) < 20000:
            sample_tokenized = [bos_token_id] + tokenizer(text, truncation=False, padding=False)['input_ids'] + [
                eos_token_id]
        else:
            # split the text into paragraphs, where each paragraph should be at least 200000 characters. Only break at "\n\n"
            paragraphs = split_into_paragraphs(text)
            tokenized_paragraphs = [tokenizer(paragraph, truncation=False, padding=False)['input_ids'] for paragraph in
                                    paragraphs]
            # concatenate all paragraphs
            concat_paragraphs = [item for sublist in tokenized_paragraphs for item in sublist]
            sample_tokenized = [bos_token_id] + concat_paragraphs + [eos_token_id]

        sample_loss_mask = [1] * len(sample_tokenized)  # compute loss for all tokens
        result = {
            "token_ids": sample_tokenized,
            "loss_mask": sample_loss_mask
        }

        debug = False
        if debug:
            debug_print(result)

        return result

    if name in ['program_books', 'textbooks']:
        process_sample_func = process_sample2  # markdown format
    else:
        process_sample_func = process_sample1  # instruct format

    # sequentially process all documents, as the process is fast enough, and to avoid race condition for stats
    for document in tqdm(all_documents):
        tokenized_document = process_sample_func(document)
        all_tokenized_documents.append(tokenized_document)
git@github.com:mtian8/Megatron-LM.git
    stats['num_samples'] = len(all_tokenized_documents)
    # print('Number of samples: ', stats['num_samples'])
    logger.info('Number of samples: %s', stats['num_samples'])

    concat_dataset = ConcatTokensDataset(
        all_tokenized_documents,
        max_length=CONCAT_LENGTH + 1,
    )

    all_concated_samples = list(concat_dataset)  # itearte through the dataset to get all samples

    # shift left to get labels
    all_sample_with_labels = []
    for sample in all_concated_samples:
        sample_with_labels = {
            'token_ids': sample['token_ids'][:-1],  # remove the last token for input
            'loss_mask': sample['loss_mask'][1:],
            'label_ids': sample['token_ids'][1:]  # remove the first token for label
        }

        if sum(sample_with_labels[
                   'loss_mask']) == 0:  # sometimes the human input can be too long, and the model response is empty after truncation
            continue

        assert len(sample_with_labels['token_ids']) == CONCAT_LENGTH
        assert len(sample_with_labels['loss_mask']) == CONCAT_LENGTH
        assert len(sample_with_labels['label_ids']) == CONCAT_LENGTH

        all_sample_with_labels.append(sample_with_labels)

    stats['num_concat_samples'] = len(all_sample_with_labels)
    stats['total_tokens'] = stats['num_concat_samples'] * CONCAT_LENGTH
    stats['total_loss_tokens'] = sum([sum(x['loss_mask']) for x in all_sample_with_labels])

    # calculate stat avergae
    if process_sample_func == process_sample1:
        stats['avg_question_length'] = stats['sum_question_tokens'] / stats['sum_quesiion_count']
        stats['avg_reply_length'] = stats['sum_reply_tokens'] / stats['sum_reply_count']
        stats['avg_reply_count_per_conversation'] = stats['sum_reply_count'] / stats['num_samples']
        stats['avg_question_count_per_conversation'] = stats['sum_quesiion_count'] / stats['num_samples']

    # print(stats)
    logger.info('Stats: %s', stats)

    # write to jsonl
    #git@github.com:mtian8/Megatron-LM.git print('Writing to jsonl...')
    logger.info('Writing to jsonl...')

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.jsonl'), 'w') as fout:
        for sample in tqdm(all_sample_with_labels):
            fout.write(json.dumps(sample) + '\n')

    # write a stat.json
    with open(os.path.join(output_dir, 'stat.json'), 'w') as fout:
        json.dump(stats, fout, indent=4)

    # print(f'Finished processing: {name}')
    logger.info('Finished processing: %s', name)

process_func_base_full = process_func_base

def prepare_tokenizer(tokenizer_path):
    global process_func_base
    process_func_base = partial(process_func_base_full, tokenizer_path=tokenizer_path)


def process_slim_orca():
    input_files = [
        'raw/SlimOrca/oo-labeled_correct.gpt4.sharegpt.jsonl'
    ]
    output_dir = 'processed/SlimOrca'
    process_func_base('SlimOrca', input_files, output_dir)


def process_html_alpaca():
    input_files = [
        'raw/html_gpt3.5_50k.json'
    ]
    output_dir = 'processed/html_alpaca'
    process_func_base('html_alpaca', input_files, output_dir)


def process_rosetta_alpaca():
    input_files = [
        'raw/rosetta_alpaca.json'
    ]
    output_dir = 'processed/rosetta_alpaca'
    process_func_base('rosetta_alpaca', input_files, output_dir)


def process_oasst1_guanaco():
    input_files = [
        'raw/oasst1-guanaco-extended-sharegpt/guanaco.sharegpt.jsonl'
    ]
    output_dir = 'processed/oasst1_guanaco'
    process_func_base('oasst1_guanaco', input_files, output_dir)


def process_sharegpt():
    input_files = [
        'raw/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json'
    ]
    output_dir = 'processed/sharegpt'
    process_func_base('sharegpt', input_files, output_dir)


def process_chatlogs():
    input_files = glob('raw/chatlogs-en-cleaned/chatlogs_v2_cleaned*.jsonl')
    output_dir = 'processed/chatlogs'
    process_func_base('chatlogs', input_files, output_dir)


def process_evol_sharegpt():
    input_files = [
        'raw/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.json'
    ]
    output_dir = 'processed/evol_sharegpt'
    process_func_base('evol_sharegpt', input_files, output_dir)


def process_codealpaca():
    input_files = [
        'raw/code_alpaca_20k.json'
    ]
    output_dir = 'processed/codealpaca'
    process_func_base('codealpaca', input_files, output_dir)


def process_evol_codealpaca1():
    input_files = [
        'raw/evol-codealpaca-v1.json'
    ]
    output_dir = 'processed/evol_codealpaca1'
    process_func_base('evol_codealpaca1', input_files, output_dir)


def process_evol_codealpaca2():
    input_files = [
        'raw/EvolInstruct-Code-80k.json'
    ]
    output_dir = 'processed/evol_codealpaca2'
    process_func_base('evol_codealpaca2', input_files, output_dir)


def process_program_books():
    input_files = [
        'raw/programming_books_llama'
    ]
    output_dir = 'processed/program_books'
    process_func_base('program_books', input_files, output_dir)


def process_textbooks():
    input_files = [
        'raw/textbooks'
    ]
    output_dir = 'processed/textbooks'
    process_func_base('textbooks', input_files, output_dir)


def process_textbooks():
    input_files = [
        'raw/textbooks'
    ]
    output_dir = 'processed/textbooks'
    process_func_base('textbooks', input_files, output_dir)


def process_sharegpt_examples():
    input_files = [
        '../examples/sharegpt_examples.json'
    ]
    output_dir = '../examples/processed/sharegpt_examples'
    process_func_base('sharegpt_examples', input_files, output_dir)


if __name__ == "__main__":
    # process_slim_orca()
    # process_html_alpaca()
    # process_rosetta_alpaca()
    # process_oasst1_guanaco()
    # process_chatlogs()
    # process_evol_sharegpt()
    # process_codealpaca()
    # process_evol_codealpaca1()
    # process_evol_codealpaca2()
    # process_program_books()
    # process_textbooks()
    # process_sharegpt()
    prepare_tokenizer(sys.argv[1])  # take tokenizer path from args
    process_slim_orca()
    process_codealpaca()
    process_evol_sharegpt()
