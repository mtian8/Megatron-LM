# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Tokenization utilities."""


import torch


from megatron.training import get_args, get_tokenizer
from .communication import broadcast_int_list, broadcast_tensor


def detokenize_generations(tokens_gpu_tensor,
                           lengths_gpu_tensor,
                           return_segments,
                           ignore_special_tokens=False):
    """Detokenize the generated tokens."""

    args = get_args()
    tokenizer = get_tokenizer()
    prompts_plus_generations = []
    if return_segments:
        prompts_plus_generations_segments = []

    if isinstance(tokens_gpu_tensor, list):
        tokens = tokens_gpu_tensor
        lengths = lengths_gpu_tensor
    else:
        tokens = tokens_gpu_tensor.cpu().numpy().tolist()
        lengths = lengths_gpu_tensor.cpu().numpy().tolist()
    for sequence_tokens, length in zip(tokens, lengths):
        sequence_tokens = sequence_tokens[:length]
        if ignore_special_tokens and args.tokenizer_type == "HuggingFaceTokenizer":
            sequence_tokens = [token for token in sequence_tokens if token not in tokenizer.special_token_id_list]
        prompts_plus_generations.append(
            tokenizer.detokenize(sequence_tokens))
        if return_segments:
            words = []
            for token in sequence_tokens:
                if args.tokenizer_type in ['SentencePieceTokenizer',
                                           'GPTSentencePieceTokenizer',
                                           'HuggingFaceTokenizer',
                                           'Llama2Tokenizer']:
                    word = tokenizer.decoder[token]
                elif args.tokenizer_type == 'TikTokenizer':
                    word = tokenizer.detokenize([token])
                elif args.tokenizer_type in ['Llama3Tokenizer', 'MistralTokenizer']:
                    word = tokenizer.decode([token])
                elif args.tokenizer_type == 'NullTokenizer':
                    word = str(token)
                else:
                    word = tokenizer.tokenizer.decoder[token]
                    word = bytearray(
                        [tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                            'utf-8', errors='replace')
                words.append(word)
            prompts_plus_generations_segments.append(words)

    if return_segments:
        return tokens, prompts_plus_generations, \
            prompts_plus_generations_segments

    return tokens, prompts_plus_generations


def tokenize_prompts(prompts=None, tokens_to_generate=None,
                     add_BOS=None, rank=0, ignore_special_tokens=False):
    """Tokenize prompts and make them avaiable on all ranks."""

    # On all ranks set to None so we can pass them to functions
    sizes_list = None
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None
    # On the specified rank, build the above.
    if torch.distributed.get_rank() == rank:
        # print("Rank 0:")
        assert prompts is not None
        assert tokens_to_generate is not None
        # Tensor of tokens padded and their unpadded length.
        prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = \
            _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS, ignore_special_tokens)
        # We need the sizes of these tensors for the boradcast
        sizes_list = [prompts_tokens_cuda_long_tensor.size(0), # Batch size
                      prompts_tokens_cuda_long_tensor.size(1)] # Sequence lenght

    # First, broadcast the sizes.
    sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=rank)
    # print("TOKENIZE", sizes_tensor, torch.distributed.get_rank())
    # Now that we have the sizes, we can boradcast the tokens
    # and length tensors.
    sizes = sizes_tensor.tolist()
    # print("Broadcast 1 at rank", torch.distributed.get_rank())
    prompts_tokens_cuda_long_tensor = broadcast_tensor(
        sizes, torch.int64, tensor=prompts_tokens_cuda_long_tensor, rank=rank)
    # print("Broadcast 2 at rank", torch.distributed.get_rank())
    prompts_length_cuda_long_tensor = broadcast_tensor(
        sizes[0], torch.int64, tensor=prompts_length_cuda_long_tensor,
        rank=rank)
    # print("Broadcast 3 at rank", torch.distributed.get_rank())



    return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor

def tokenize_prompts_on_one_rank(prompts=None, tokens_to_generate=None,
                     add_BOS=None, rank=0, ignore_special_tokens=False):
    """Tokenize prompts and make them avaiable on all ranks."""

    # On all ranks set to None so we can pass them to functions
    sizes_list = None
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None
    # On the specified rank, build the above.
    assert torch.distributed.get_rank() == rank, "tokenize_prompts_on_one_rank should only be called on rank 0."
    # print("Rank 0:")
    assert prompts is not None
    assert tokens_to_generate is not None
    # Tensor of tokens padded and their unpadded length.
    prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = \
        _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS, ignore_special_tokens)
    # We need the sizes of these tensors for the boradcast
    sizes_list = [prompts_tokens_cuda_long_tensor.size(0), # Batch size
                  prompts_tokens_cuda_long_tensor.size(1)] # Sequence lenght

# First, broadcast the sizes.

# print("TOKENIZE", sizes_tensor, torch.distributed.get_rank())
# Now that we have the sizes, we can boradcast the tokens
# and length tensors.
    sizes = sizes_list
# print("Broadcast 1 at rank", torch.distributed.get_rank())

# print("Broadcast 3 at rank", torch.distributed.get_rank())



    return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor


def _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS, ignore_special_tokens=False):
    """Given a set of prompts and number of tokens to generate:
        - tokenize prompts
        - set the sequence length to be the max of length of prompts
          plus the number of tokens we would like to generate
        - pad all the sequences to this length so we can convert them
          into a 2D tensor.
    """
    # print("Tokenize prompts and batch")
    # Tokenize all the prompts.
    args = get_args()
    tokenizer = get_tokenizer()
    if hasattr(tokenizer, 'eod'):
        eod_token = tokenizer.eod
    elif hasattr(tokenizer, 'eos_id'):
        eod_token = tokenizer.eos_id
    else:
        raise AttributeError('No eod token found in Tokenizer')
    if add_BOS:
        # print("Adding BOS")
        prompts_tokens = [[eod_token] + tokenizer.tokenize(prompt)
                          for prompt in prompts]
    else:
        prompts_tokens = [tokenizer.tokenize(prompt) for prompt in prompts]
        if ignore_special_tokens and args.tokenizer_type == "HuggingFaceTokenizer":
            prompts_tokens = [[token for token in prompt_tokens if token not in tokenizer.special_token_id_list]
                              for prompt_tokens in prompts_tokens]
        # print("Did not add BOS; first token is", prompts_tokens[0][0], "; tokenizer.eod is", eod_token)
        # for special_token in ["cls", "sep", "pad", "eod", "bos", "eos", "mask"]:
            # try:
            #     print(f"{special_token}: {getattr(tokenizer, special_token)}")
            # except Exception as e:
            #     print(f"Error getting {special_token}: {e}")

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.
    prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
    # Get the max prompts length.
    # print(prompts_length, "Prompts length")
    max_prompt_len = max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = max_prompt_len + tokens_to_generate
    # print("Tokens to generate", tokens_to_generate, "samples length", samples_length)
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
        padding_size = samples_length - prompt_length
        prompt_tokens.extend([eod_token] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.long, device='cuda')
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.long, device='cuda')
    # print("Finished tokenize prompts and batch:", prompts_tokens_tensor.size(), prompts_length_tensor)
    # if tokens_to_generate != 0:
    #     input()
    return prompts_tokens_tensor, prompts_length_tensor
