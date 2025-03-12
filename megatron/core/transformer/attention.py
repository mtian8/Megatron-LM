# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import os.path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import version
from typing import Union

import torch
from pkg_resources import packaging

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

from .enums import AttnMaskType
from .transformer_config import TransformerConfig
from ..focused_positions import FocusedPositions
from ...inference.text_generation.communication import broadcast_int_list

try:
    import transformer_engine

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

if HAVE_TE:
    from megatron.core.transformer.custom_layers.transformer_engine import SplitAlongDim
else:
    SplitAlongDim = None

from megatron.core.utils import debug, use_debug, change_debug


@dataclass
class SelfAttentionSubmodules:
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class CrossAttentionSubmodules:
    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class Attention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
            layer_number: int,
            attn_mask_type: AttnMaskType,
            attention_type: str,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

    def _checkpointed_attention_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb=None,
            attn_mask_type=None,
            packed_seq_params=None,
            extra_kwargs=None
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
                extra_kwargs=extra_kwargs
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type,
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_length, batch_size, dtype):
        """Allocate memory to store kv cache during inference."""

        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def _adjust_key_value_for_inference(self, inference_params, key, value, rotary_pos_emb):
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params, as well as
        adjusted rotary_pos_emb.

        Returns a tuple: (key, value, rotary_pos_emb)

        """
        attn_mask_type = self.attn_mask_type
        if inference_params is None:
            return key, value, rotary_pos_emb, attn_mask_type

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_length = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, key.dtype
            )
            inference_value_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, value.dtype
            )
            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory,
                inference_value_memory,
            )
        else:
            # Get the pre-allocated buffers for this layer
            inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[
                self.layer_number
            ]

        if inference_params.sequence_len_offset > 0:
            # This should mean that we are past the prompt forward_step
            # and so we need to turn off masking
            attn_mask_type = AttnMaskType.no_mask

        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
        inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value
        key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
        value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

        # adjust the key rotary positional embedding
        if rotary_pos_emb is None:
            return key, value, rotary_pos_emb, attn_mask_type

        # ######### TEST ####### #
        # ORACLE POSITIONAL EMBEDDING
        rotary_pos_emb = self._modify_rotary_pos_emb_from_oracle_pattern(key, value, rotary_pos_emb, inference_params)

        q_pos_emb, k_pos_emb = rotary_pos_emb
        q_pos_emb = q_pos_emb[sequence_start:sequence_end, :, :, :]
        k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
        rotary_pos_emb = (q_pos_emb, k_pos_emb)

        return key, value, rotary_pos_emb, attn_mask_type

    @abstractmethod
    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def _modify_rotary_pos_emb_from_oracle_pattern(self, key, value, rotary_pos_emb, inference_params):
        """Modifies the rotary positional embedding based on the oracle pattern.

        Args:
            rotary_pos_emb (tuple): The rotary positional embedding.
            inference_params (InferenceParams): The inference parameters.

        Returns:
            tuple: The modified rotary positional embedding.
        """
        if inference_params is None:
            return rotary_pos_emb

        if not hasattr(inference_params, "other_kwargs"):
            return rotary_pos_emb

        # retrieve focused positions
        # ignore oracle mode for distance_between_positions
        distance_between_positions = inference_params.other_kwargs.get("distance_between_positions", 0)
        oracle_positions = inference_params.other_kwargs.get("oracle_positions", None)
        pattern_mode = inference_params.other_kwargs.get("pattern_mode", "off")

        if distance_between_positions:  # if distance is given, then ignore pattern mode, as if it were "off"
            focused_positions = FocusedPositions(1, oracle_positions, 0)
        else:
            if pattern_mode == "off":  # no pattern and no distance, no modification
                return rotary_pos_emb
            if pattern_mode == "oracle":   # oracle mode: pattern_id given
                if not oracle_positions:
                    return rotary_pos_emb
                focused_positions = self.core_attention.get_focused_positions_from_oracle_positions(oracle_positions)

            # pattern_mode == dynamic: infer from extracted_pattern_id
            else:
                if not inference_params.other_kwargs.get("dynamic_pattern_id", None):
                    return rotary_pos_emb
                dynamic_pattern_id = inference_params.other_kwargs["dynamic_pattern_id"]
                if len(dynamic_pattern_id) == 0:
                    return rotary_pos_emb
                dynamic_pattern_id = dynamic_pattern_id[0]
                focused_positions = self.core_attention.get_focused_positions_from_pattern_id(dynamic_pattern_id)

        if focused_positions is None or len(focused_positions.intervals) == 0:  # full attention
            return rotary_pos_emb

        q_pos_emb, k_pos_emb = rotary_pos_emb
        sequence_end = key.size(0)
        maximum_sequence_end = inference_params.max_sequence_length

        if distance_between_positions:
            intervals = focused_positions.get_all_positions(seq_len=sequence_end, attention_sink=False,
                                                            merge_overlapped_intervals=False)
        else:
            intervals = focused_positions.get_all_positions(seq_len=sequence_end, attention_sink=True)
        last_start = 0
        embeddings_source = [(0, 0) for _ in intervals]
        new_embeddings_k = []
        new_embeddings_q = []
        debug_str = ""
        src_end = 0
        # k_pos_emb_128 = torch.ones_like(k_pos_emb) * k_pos_emb[128:129, :, :, :]
        # q_pos_emb_128 = torch.ones_like(q_pos_emb) * q_pos_emb[128:129, :, :, :]
        if torch.distributed.get_rank() == 0 and self.layer_number == 3:
            print(
                f"[rank {torch.distributed.get_rank()}] Intervals: {intervals}, Before: q_pos_emb: {q_pos_emb.shape}, k_pos_emb: {k_pos_emb.shape}. Max seq len: {inference_params.max_sequence_length}")

        for idx in range(len(intervals)):
            start, end = intervals[idx]
            if idx > 0:
                new_embeddings_k.append(k_pos_emb[last_start: start])
                new_embeddings_q.append(q_pos_emb[last_start: start])
                # new_embeddings_k.append(k_pos_emb[src_end: src_end + 1].expand(start - last_start, -1, -1, -1))
                # new_embeddings_q.append(q_pos_emb[src_end: src_end + 1].expand(start - last_start, -1, -1, -1))
                debug_str += f"[{src_end}:{src_end + 1} (*{start - last_start})]"

            if src_end == 128:
                src_end = start
            src_start = src_end
            src_end = src_end + end - start
            new_embeddings_k.append(k_pos_emb[src_start: src_end])
            new_embeddings_q.append(q_pos_emb[src_start: src_end])

            # new_embeddings_q.append(q_pos_emb[start: end])
            debug_str += f"[{src_start}:{src_end}]"
            if distance_between_positions:
                src_end += distance_between_positions
            last_start = end
        new_embeddings_k.append(k_pos_emb[last_start:maximum_sequence_end])
        new_embeddings_q.append(q_pos_emb[last_start:maximum_sequence_end])
        debug_str += f"[{last_start}:{maximum_sequence_end}]"
        k_pos_emb = torch.cat(new_embeddings_k, dim=0)
        q_pos_emb = torch.cat(new_embeddings_q, dim=0)
        if torch.distributed.get_rank() == 0 and self.layer_number == 3:
            print(
                f"[rank {torch.distributed.get_rank()}] Intervals: {intervals}, !{debug_str}, After: q_pos_emb: {q_pos_emb.shape}, k_pos_emb: {k_pos_emb.shape}")
        return q_pos_emb, k_pos_emb


    def forward(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            inference_params=None,
            rotary_pos_emb=None,
            packed_seq_params=None,
    ):
        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        extra_kwargs = {}
        extra_kwargs["prefill"] = inference_params.sequence_len_offset == 0
        if extra_kwargs["prefill"]:
            if self.layer_number == 1:
                inference_params.other_kwargs["tokens_generated"] = 0
                inference_params.other_kwargs["dynamic_pattern_id"] = []
        else:
            if self.layer_number == 1:
                inference_params.other_kwargs["tokens_generated"] += key.size(0)
        extra_kwargs["tokens_generated"] = inference_params.other_kwargs["tokens_generated"]
        extra_kwargs["dynamic_pattern_id"] = inference_params.other_kwargs["dynamic_pattern_id"]
        extra_kwargs["pattern_mode"] = inference_params.other_kwargs["pattern_mode"]
        extra_kwargs["oracle_positions"] = inference_params.other_kwargs.get("oracle_positions", None)
        extra_kwargs["distance_between_positions"] = inference_params.other_kwargs.get("distance_between_positions", 0)
        extra_kwargs["attention_save_file"] = inference_params.other_kwargs.get("attention_save_file", "")

        # for i in range(1):
        #     # l1 = [1]
        #     # broadcast_int_list(1, int_list=l1)
        #     if self.layer_number < 3 and extra_kwargs["tokens_generated"] < 5 and torch.distributed.get_rank() == 0:
        #         print(f"Layer {self.layer_number} Rank {torch.distributed.get_rank()} Token {extra_kwargs['tokens_generated']}: Hidden={hidden_states.sum()}")
        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        # !!!!!!!!!!!!!!!!!!!!!! WARNING: EXPERIMENTAL #########################
        # Modify unpatterned queries, keys, values to space

        if self.layer_number < 3 and torch.distributed.get_rank() == 0 and extra_kwargs['tokens_generated'] < 5:
            print(
                f"Layer {self.layer_number} Rank {torch.distributed.get_rank()} Token {extra_kwargs['tokens_generated']} Pre rope query: {query.max()}\n", end=""
                )
            print(
                f"Layer {self.layer_number} Rank {torch.distributed.get_rank()} Token {extra_kwargs['tokens_generated']} Pre rope key: { key.max()} \n", end=""
               )
            print(
                f"Layer {self.layer_number} Rank {torch.distributed.get_rank()} Token {extra_kwargs['tokens_generated']} Pre rope value: {value.max()}\n", end="",
                )

        if self.layer_number == 1 and extra_kwargs["pattern_mode"] == "oracle" and extra_kwargs["distance_between_positions"] == 1437:  # magic number!
            oracle_positions = extra_kwargs["oracle_positions"]
            oracle_positions = [[start+1, end+1] for start, end in oracle_positions]
            focused_positions = FocusedPositions(1, oracle_positions, extra_kwargs["tokens_generated"] + 235)
            intervals = focused_positions.get_all_positions(seq_len=key.size(0), attention_sink=True,  # the first token is 1
                                                            merge_overlapped_intervals=True)
            # if torch.distributed.get_rank() == 0:
            #     print(f"[R {torch.distributed.get_rank()} token {extra_kwargs['tokens_generated']}] EXPERIMENTAL: {intervals}, seq_len: {key.size(0)}, {value.size(0)}, {query.size(0)}\n", end="")
            previous_end = 0
            new_queries = []
            new_keys = []
            new_values = []
            for start, end in intervals:
                new_keys.append(key[1:2].expand(start - previous_end, -1, -1, -1))
                new_keys.append(key[start:end])
                new_values.append(value[1:2].expand(start - previous_end, -1, -1, -1))
                new_values.append(value[start:end])
                if inference_params.other_kwargs["tokens_generated"] == 0:
                    new_queries.append(query[1:2].expand(start - previous_end, -1, -1, -1))
                    new_queries.append(query[start:end])
                    if previous_end < start:
                        hidden_states[previous_end:start] = hidden_states[1:2].expand(start - previous_end, -1, -1)  # substitute directly
                previous_end = end

            if previous_end < key.size(0):
                new_keys.append(key[previous_end:])
                new_values.append(value[previous_end:])
                if inference_params.other_kwargs["tokens_generated"] == 0:
                    new_queries.append(query[previous_end:])
            if inference_params.other_kwargs["tokens_generated"] > 0:
                new_queries.append(query)



            key = torch.cat(new_keys, dim=0)
            value = torch.cat(new_values, dim=0)
            query = torch.cat(new_queries, dim=0)
            # if torch.distributed.get_rank() == 0:
            #     print(f"[R {torch.distributed.get_rank()}] ~~~~EXPERIMENTAL: {intervals}, seq_len: {key.size(0)}, {value.size(0)}, {query.size(0)}\n", end="")
        # !!!!!!!!!!!!!!!!!!!!!! END OF WARNING: EXPERIMENTAL #########################

        # if self.layer_number < 3 and torch.distributed.get_rank() == 0 and extra_kwargs['tokens_generated'] < 5:
        #     print(f"Layer {self.layer_number} Rank 0 Token {extra_kwargs['tokens_generated']}: {query.size(0)}, {key.size(0)}, {value.size(0)}")
        #     def print_key(tensor):
        #         print(f"{tensor[:, 0, 0, :2].reshape(-1).cpu().to(float).numpy().tolist()}\n", end="")
        #     print_key(key[:8])
        #     q_start_position = key.size(0) - extra_kwargs["tokens_generated"] - 235
        #     print_key(key[q_start_position - 8: q_start_position + 8])
        #     print(f"{q_start_position}.")

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb




            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(
                query,
                q_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
            )
            key = apply_rotary_pos_emb(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
            )
            debug("q pos emb: ", q_pos_emb[..., :6])
            debug("k pos emb: ", k_pos_emb[..., :6])
            for i in range(1):
                # l1 = [1]
                # broadcast_int_list(1, int_list=l1)
                if self.layer_number < 3 and torch.distributed.get_rank() == 0 and extra_kwargs['tokens_generated'] < 5:

                    print(f"Layer {self.layer_number} Rank {torch.distributed.get_rank()} Token {extra_kwargs['tokens_generated']} Post rope query:  {query.max()}\n", end="")
                    print(f"Layer {self.layer_number} Rank {torch.distributed.get_rank()} Token {extra_kwargs['tokens_generated']} Post rope key:  {key.max()}\n", end="")
                    print(f"Layer {self.layer_number} Rank {torch.distributed.get_rank()} Token {extra_kwargs['tokens_generated']} Post rope value:  {value.max()}\n", end="")
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        if hasattr(inference_params, "other_kwargs") and inference_params.other_kwargs.get("pattern_id", None):
            extra_kwargs["pattern_id"] = inference_params.other_kwargs["pattern_id"]



        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
                extra_kwargs=extra_kwargs,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
                extra_kwargs=extra_kwargs,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        debug("Core attention out", core_attn_out[..., :6])
        # =================
        # Output. [sq, b, h]
        # =================
        # if self.layer_number < 3 and torch.distributed.get_rank() == 0 and extra_kwargs['tokens_generated'] < 5:
        #     print(
        #         f"Layer {self.layer_number} Rank {torch.distributed.get_rank()} Token {extra_kwargs['tokens_generated']} Attn Out: ",
        #         core_attn_out.sum())
        output, bias = self.linear_proj(core_attn_out)

        debug("Linear proj Output: ", output[..., :6])
        return output, bias


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def run_realtime_tests(self):
        """Performs a consistency check.

        This function makes sure that tensors across devices are the same during an experiment.
        This is often not guaranteed to be so because of silent hardware failures (eg, memory
        corruption loading a checkpoint, network traffic corruption encountered during data transmission).

        (TODO) In the future, more tensors should be checked across the training run and
        checked every X iterations. This is left for future work. Equality of tensors is probably not
        required; transmitting hashes is sufficient."""

        if not self.config.qk_layernorm:
            return

        # check that all tensor parallel and data parallel ranks have the same
        # Q & K layernorm parameters.
        rank = get_data_parallel_rank()
        inputs = torch.stack(
            [
                self.q_layernorm.weight.data,
                self.q_layernorm.bias.data,
                self.k_layernorm.weight.data,
                self.k_layernorm.bias.data,
            ]
        )
        dp_list = [torch.empty_like(inputs) for _ in range(get_data_parallel_world_size())]
        dp_list[rank] = inputs
        torch.distributed.all_gather(dp_list, inputs, group=get_data_parallel_group())

        def _compare(srcs, tgts, names, parallelism):
            assert len(srcs) == len(tgts) == len(names)
            for src, tgt, name in zip(srcs, tgts, names):
                assert torch.all(
                    src == tgt
                ), f"Discrepancy between {name} in {parallelism} ranks {i} and {rank}. Diff: {torch.norm(src - tgt)}"

        for i, dp in enumerate(dp_list):
            q_w, q_b, k_w, k_b = torch.unbind(dp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "DP",
            )

        rank = get_tensor_model_parallel_rank()
        tp_list = [torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())]
        tp_list[rank] = inputs
        torch.distributed.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())

        for i, tp in enumerate(tp_list):
            q_w, q_b, k_w, k_b = torch.unbind(tp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "TP",
            )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        debug("Mixed QKV: ", mixed_qkv[..., :6])
        debug("Mixed QKV shape: ", mixed_qkv.size())
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        debug("Mixed QKV reshaped: ", mixed_qkv.shape)
        split_arg_list = [
            (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        # input()
        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(
                mixed_qkv,
                3,
                split_arg_list,
            )
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(
                mixed_qkv,
                split_arg_list,
                dim=3,
            )

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()
        debug("Query: ", query[..., :6])
        debug("Key:   ", key[..., :6])
        debug("Value: ", value[..., :6])
        return query, key, value


class CrossAttention(Attention):
    """Cross-attention layer class

    Cross-attention layer takes input with size [s, b, h] and context with size
    [s, b, h] and returns output of the same size.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: CrossAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
        )

        if self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Group query attention is not currently supported in cross attention."
            )
        assert self.query_projection_size == self.kv_projection_size

        self.linear_q = build_module(
            submodules.linear_q,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv = build_module(
            submodules.linear_kv,
            self.config.hidden_size,
            2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv.size()[:-1] + (
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query, _ = self.linear_q(hidden_states)

        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        return query, key, value
