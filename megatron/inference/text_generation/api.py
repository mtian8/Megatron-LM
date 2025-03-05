# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Inference API."""


import torch

from megatron.core import mpu
from .communication import broadcast_float_list, broadcast_int_list
from .generation import (
        generate_tokens_probs_and_return_on_first_stage,
        score_and_return_on_first_stage,
        beam_search_and_return_on_first_stage)
from .tokenization import (
    tokenize_prompts,
    detokenize_generations)
from .forward_step import ForwardStep
from transformer_engine.pytorch.attention import check_set_window_size

def generate_and_post_process(model,
                              forward_step=ForwardStep,
                              prompts=None,
                              tokens_to_generate=0,
                              return_output_log_probs=False,
                              top_k_sampling=0,
                              top_p_sampling=0.0,
                              top_p_decay=0.0,
                              top_p_bound=0.0,
                              temperature=1.0,
                              add_BOS=False,
                              use_eod_token_for_early_termination=True,
                              stop_on_double_eol=False,
                              stop_on_eol=False,
                              prevent_newline_after_colon=False,
                              random_seed=-1,
                              return_logits=False,
                              ignore_special_tokens=False,
                              oracle_positions=None,
                              oracle_mode="off",
                              distance_between_positions=0,
                                attention_save_file=""
                              ):
    """Run inference and post-process outputs, i.e., detokenize,
    move to cpu and convert to list."""

    # Main inference.
    tokens, lengths, output_log_probs, logits = generate(
        model,
        forward_step=forward_step,
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        return_output_log_probs=return_output_log_probs,
        top_k_sampling=top_k_sampling,
        top_p_sampling=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        add_BOS=add_BOS,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        random_seed=random_seed,
        oracle_positions=oracle_positions,
        oracle_mode=oracle_mode,
        distance_between_positions=distance_between_positions,
        attention_save_file=attention_save_file
    )

    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        tokens, prompts_plus_generations, prompts_plus_generations_segments = \
            detokenize_generations(tokens, lengths, True, ignore_special_tokens)

        if return_output_log_probs:
            output_log_probs = output_log_probs.cpu().numpy().tolist()
            for i, (prob, seg) in enumerate(zip(output_log_probs, prompts_plus_generations_segments)):
                output_log_probs[i] = prob[:len(seg)-1]

        if return_logits:
            # assert(tokens_to_generate == 0)
            assert(mpu.get_pipeline_model_parallel_world_size() == 1)
            return prompts_plus_generations, prompts_plus_generations_segments, \
            output_log_probs, tokens, logits
        else:
            return prompts_plus_generations, prompts_plus_generations_segments, \
            output_log_probs, tokens, None

    return None

def generate(model,
             forward_step=None,
             prompts=None,
             tokens_to_generate=0,
             return_output_log_probs=False,
             top_k_sampling=0,
             top_p_sampling=0.0,
             top_p_decay=0.0,
             top_p_bound=0.0,
             temperature=1.0,
             add_BOS=False,
             use_eod_token_for_early_termination=True,
             stop_on_double_eol=False,
             stop_on_eol=False,
             prevent_newline_after_colon=False,
             random_seed=-1,
             oracle_positions=None,
             oracle_mode="off",
             distance_between_positions=0,
             attention_save_file="",
             ):
    """Given prompts and input parameters, run inference and return:
       tokens: prompts plus the generated tokens.
       lengths: length of the prompt + generations. Note that we can
           discard tokens in the tokens tensor that are after the
           corresponding length.
       output_log_probs: log probs of the tokens.
    """

    # Make sure input params are avaialble to all ranks.
    # print(f"[Rank {torch.distributed.get_rank()}] tokens_to_generate: {tokens_to_generate}")
    values = [tokens_to_generate,
              return_output_log_probs,
              top_k_sampling, top_p_sampling, top_p_decay, top_p_bound,
              temperature, add_BOS, use_eod_token_for_early_termination,
              stop_on_double_eol,
              stop_on_eol,
              prevent_newline_after_colon,
              random_seed]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k_sampling = int(values_float_tensor[2].item())
    top_p_sampling = values_float_tensor[3].item()
    top_p_decay = values_float_tensor[4].item()
    top_p_bound = values_float_tensor[5].item()
    temperature = values_float_tensor[6].item()
    add_BOS = bool(values_float_tensor[7].item())
    use_eod_token_for_early_termination = bool(values_float_tensor[8].item())
    stop_on_double_eol = bool(values_float_tensor[9].item())
    stop_on_eol = bool(values_float_tensor[10].item())
    prevent_newline_after_colon = bool(values_float_tensor[11].item())
    random_seed = int(values_float_tensor[12].item())
    values_int_tensor = [0 for _ in range(33)]  # only support num_oracle_positions <= 16
    if oracle_positions is not None:  # [batch_size, num_oracle_positons, 2]; only support batch size 1
        for i, instance in enumerate(oracle_positions[0]):
            values_int_tensor[i * 2 + 1] = instance[0]
            values_int_tensor[i * 2 + 2] = instance[1]
        values_int_tensor[0] = len(oracle_positions[0])
    values_int_tensor = broadcast_int_list(len(values_int_tensor), int_list=values_int_tensor)
    if values_int_tensor[0] > 0:
        oracle_positions = [[]]
        for i in range(values_int_tensor[0]):
            oracle_positions[0].append([values_int_tensor[i * 2 + 1].item(), values_int_tensor[i * 2 + 2].item()])
    else:
        oracle_positions = None

    values_int_tensor[0] = distance_between_positions
    values_int_tensor[1] = 0 if oracle_mode == "off" else (1 if oracle_mode == "debug" else 2)
    values_int_tensor = broadcast_int_list(2, int_list=values_int_tensor[:2])
    distance_between_positions = values_int_tensor[0]
    if values_int_tensor[1] == 0:
        oracle_mode = "off"
    elif values_int_tensor[1] == 1:
        oracle_mode = "debug"
    else:
        oracle_mode = "on"

    print(f"[Rank {torch.distributed.get_rank()}] oracle_positions: {oracle_positions}")

    string_list = [attention_save_file]
    torch.distributed.broadcast_object_list(string_list, 0)
    attention_save_file = string_list[0]

    if random_seed != -1:
        torch.random.manual_seed(random_seed)

    # Tokenize prompts and get the batch.
    # Note that these tensors are broadcaseted to all ranks.
    if torch.distributed.get_rank() == 0:
        assert prompts is not None


    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS, add_space=True)

    if tokens_to_generate == 0:
        return score_and_return_on_first_stage(
            model, context_tokens_tensor, context_length_tensor)
    # print(f"[Rank {torch.distributed.get_rank()}] generate_tokens_probs_and_return_on_first_stage")
    # Main inference function.
    # Note that the outputs are available on the first stage.
    return generate_tokens_probs_and_return_on_first_stage(
        model, forward_step, context_tokens_tensor, context_length_tensor,
        return_output_log_probs=return_output_log_probs,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        oracle_positions=oracle_positions,
        oracle_mode=oracle_mode,
        distance_between_positions=distance_between_positions,
        attention_save_file=attention_save_file
    )

def beam_search_and_post_process(model,
                                 forward_step=ForwardStep,
                                 prompts=None,
                                 tokens_to_generate=0,
                                 beam_size=0,
                                 add_BOS=False,
                                 stop_token=50256,
                                 num_return_gen=1,
                                 length_penalty=1,
                                 prevent_newline_after_colon=False,
                                 ignore_special_tokens=False):
    """Run beam search and post-process outputs, i.e., detokenize,
    move to cpu and convert to list."""

    # Main inference.
    tokens, scores = beam_search(model,
                                 forward_step=forward_step,
                                 prompts=prompts,
                                 tokens_to_generate=tokens_to_generate,
                                 beam_size=beam_size,
                                 add_BOS=add_BOS,
                                 stop_token=stop_token,
                                 num_return_gen=num_return_gen,
                                 length_penalty=length_penalty,
                                 prevent_newline_after_colon=prevent_newline_after_colon)
    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        lengths = tokens.size(1)*torch.ones(beam_size, dtype=torch.int64, device=torch.cuda.current_device())
        tokens, prompts_plus_generations, prompts_plus_generations_segments = detokenize_generations(tokens, lengths, True, ignore_special_tokens)
        scores = scores.cpu().numpy().tolist()
        return prompts_plus_generations, prompts_plus_generations_segments, scores

    return None

def beam_search(model, forward_step, prompts=None, tokens_to_generate=0, beam_size=0, add_BOS=False, stop_token=50256, num_return_gen=1, length_penalty=1, prevent_newline_after_colon=False):
    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate,
              beam_size,
              add_BOS,
              stop_token,
              num_return_gen,
              length_penalty,
              prevent_newline_after_colon]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    beam_size = int(values_float_tensor[1].item())
    add_BOS = bool(values_float_tensor[2].item())
    stop_token = int(values_float_tensor[3].item())
    num_return_gen = int(values_float_tensor[4].item())
    length_penalty = values_float_tensor[5].item()
    prevent_newline_after_colon = values_float_tensor[6].item()

    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)

    return beam_search_and_return_on_first_stage(model, forward_step, context_tokens_tensor, context_length_tensor,
            beam_size, stop_token=stop_token, num_return_gen=num_return_gen, length_penalty=length_penalty,
            prevent_newline_after_colon=prevent_newline_after_colon)

def modify_window_size(model, window_size="Not rank 0"):
    """Modify the window size of the model."""
    _window_size = window_size
    if window_size is None:
        window_size = (-1, 0)
    elif window_size == "Not rank 0":
        window_size = [0 for _ in range(33)]
    values = [-1 for _ in range(33)]

    if _window_size != "Not rank 0":
        for i in range(len(window_size)):
            if window_size[i] is not None:
                values[i + 1] = window_size[i]
        values[0] = len(window_size)
    print("Before sync", values)

    values_int_tensor = broadcast_int_list(len(values), int_list=values)
    window_size = [x.item() for x in values_int_tensor]
    window_size_len = window_size[0]
    window_size = window_size[1: window_size_len + 1]
    print("After sync", window_size)

    for i, layer in enumerate(model._modules["module"].decoder.layers):
        attn = layer.self_attention.core_attention
        current_window_size = tuple(window_size)
        if len(current_window_size) != 2:
            if window_size[i] is None:
                current_window_size = (-1, 0)
            else:
                current_window_size = (window_size[i], 0)
        print(f"Layer {i}: {current_window_size}")
        attn.window_size = check_set_window_size(attn.attn_mask_type, current_window_size)
    return