import os
import sys

from transformers import AutoTokenizer, AutoModel

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict,
        tokenizer,
        model,
):
    """Resize tokenizer and embedding by adding special tokens and averaging the embeddings of the new tokens.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    print("Original vocab size: ", tokenizer.vocab_size)
    print(tokenizer.special_tokens_map)
    print(special_tokens_dict)
    tokenizer.add_special_tokens({"additional_special_tokens": []})  # a bug in huggingface tokenizers
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Added {num_new_tokens} new tokens to the tokenizer.")
    # print all the special tokens
    print(tokenizer.special_tokens_map)
    if num_new_tokens > 0:
        model.resize_token_embeddings(tokenizer.vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg



def update_special_tokens(input_path, output_path, special_tokens_dict):

    tokenizer = AutoTokenizer.from_pretrained(input_path,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(input_path,
                                      trust_remote_code=True)
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)

if __name__ == '__main__':
    special_tokens_dict = {"additional_special_tokens": ['<|sys_start|>', '<|sys_end|>', '<|im_start|>', '<|im_end|>']}
    input_path = "../../models/Llama-2-7b-hf"
    output_path = "../../models/Llama-2-7b-hf-extra-tokens"
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
    update_special_tokens(input_path, output_path, special_tokens_dict)