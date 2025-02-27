import dataclasses
import math
import os
from functools import lru_cache
from importlib.metadata import version
from typing import Callable, Optional, Tuple

import torch
from pkg_resources import packaging
from torch import Tensor

from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.focused_positions import FocusedPositions, round_interval_by_block

import customized_dkernel
from customized_dkernel import SparseAttention, LocalStrideSparseAttention
from customized_dkernel.utils import multiple_of, get_sparse_attn_mask, dense_to_crow_col
import triton
import warnings

from megatron.inference.text_generation.communication import broadcast_int_list


def _get_extra_dkernel_kwargs(config: TransformerConfig):
    pass


def condition_init_method(config, init_method):
    pass


class DKernelSparseAttention(SparseAttention):
    def __init__(
            self,
            config: TransformerConfig,
            sparse_pattern,
            layer_number: int,
            attention_mask_type: AttnMaskType,
            attention_type: str,
            attention_dropout: float = None,

    ):
        # ...
        self.config = config
        self.dkernel_forward_mask_type = False
        self.qkv_format: str = "sbhd"  # the default input format for Megatron
        self.seq_dim = 1  # bshd, the default format for SparseAttention
        # ...

        extra_kwargs = {}

        super().__init__(
            block_size=self.config.sparse_block_size,
            sparse_pattern=sparse_pattern,
            # *,
            seq_dim=self.seq_dim,
            block_m=self.config.sparse_block_m,
            block_n=self.config.sparse_block_n,
            **extra_kwargs
        )

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attention_mask: Tensor,
            attn_mask_type: AttnMaskType,
            packed_seq_params: PackedSeqParams = None,
    ):
        packed_seq_params = (
            dataclasses.asdict(packed_seq_params) if packed_seq_params is not None else {}
        )

        if self.config.apply_rope_fusion:
            self.qkv_format = 'bshd'

        qkv_format = packed_seq_params.get("qkv_format", self.qkv_format)

        assert qkv_format in ["bshd", "sbhd"], f"Unsupported qkv_format: {qkv_format}"

        if qkv_format == 'sbhd':
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
            # In PyTorch, the following two tensors are in fact the same:
            #   Tensor with shape (1, S, H, D) and stride (S*H*D, H*D, D, 1)
            #   Tensor with shape (1, S, H, D) and stride (H*D, H*D, D, 1)
            # Stride for a dimension that is 1 has no meaning, so tensors created two different ways
            # can have same shape but different strides.
            # We unify them to the first one to pass the stride check in TE
            if value.shape == key.shape and value.shape[1] == 1 and value.stride() != key.stride():
                value = value.as_strided(value.shape, key.stride())

        packed_seq_params = {
            "cu_seqlen_k": None,
            "cu_seqlen_q": None,
        }

        if self.te_forward_mask_type:
            if attn_mask_type == AttnMaskType.padding:
                attention_mask = attention_mask.to(torch.bool)
                # get left paddings
                left_paddings = attention_mask.argmax(dim=-1).view(-1)
                # get right paddings
                right_paddings = attention_mask.flip(dims=[-1]).argmax(dim=-1).view(-1)
                if left_paddings.sum() > 0:
                    packed_seq_params["left_paddings"] = left_paddings
                if right_paddings.sum() > 0:
                    packed_seq_params["seqlens"] = attention_mask.shape[-1] - right_paddings

        core_attn_out = super().forward(
            query,
            key,
            value,
            sm_scale=None,
            # *,
            **packed_seq_params
        )

        if self.config.apply_rope_fusion and qkv_format == "bshd":
            return core_attn_out.transpose(0, 1)
        return core_attn_out


class SparseAttentionForTP(SparseAttention):
    def __init__(self,
                 block_size: int,
                 sparse_pattern: Tensor,
                 *,
                 seq_dim: Optional[int] = None,
                 block_m: Optional[int] = None,
                 block_n: Optional[int] = None,
                 tp_size: int = 1,
                 tp_group: Optional[torch.distributed.ProcessGroup] = None,
                 **kwargs):
        super().__init__(block_size, sparse_pattern,
                         seq_dim=seq_dim, block_m=block_m, block_n=block_n,
                         **kwargs)
        # tensor parallelism
        self.tp_size = tp_size
        self.tp_group = tp_group


def get_tp_info(self, tp_size, tp_group, num_heads, num_kv_heads):
    """
        usage:
        tp_rank, num_heads_per_partition, num_kv_heads_per_partition, partition_head_l, partition_head_u =\
            self.get_tp_info(tp_size, tp_group, num_heads, num_kv_heads)
    """
    tp_rank = torch.distributed.get_rank(group=tp_group)
    # print(f"[Layer {self.layer_number}][TPRank {tp_rank}] In LocalStride: tp_size={tp_size},"
    #       f" num_heads={num_heads}, num_kv_heads={num_kv_heads}")

    assert num_heads % tp_size == 0, "num_attention_heads must be divisible by tp_size."
    num_heads_per_partition = num_heads // tp_size

    num_kv_heads = num_kv_heads or num_heads
    num_kv_heads_per_partition = num_kv_heads // tp_size

    partition_head_l = tp_rank * num_heads_per_partition
    partition_head_u = (tp_rank + 1) * num_heads_per_partition

    return tp_rank, num_heads_per_partition, num_kv_heads_per_partition, partition_head_l, partition_head_u


@lru_cache(maxsize=8)
class LocalStrideSparseAttentionForTP(SparseAttentionForTP):
    def __init__(self,
                 num_heads: int,
                 max_seq_len: int,
                 block_size: int,
                 local_blocks: int,
                 vert_stride: int,
                 *,
                 homo_head: bool = False,
                 num_dense_heads: int = 0,
                 num_kv_heads: Optional[int] = None,
                 active_head_range: Optional[Tuple[int, int]] = None,
                 head_sliding_offset: int = 0,
                 block_m: Optional[int] = None,
                 block_n: Optional[int] = None,
                 tp_size: int = 1,
                 tp_group: Optional[torch.distributed.ProcessGroup] = None,
                 layer_number: int = None,
                 **kwargs
                 ):

        # tensor parallelism
        self.layer_number = layer_number
        tp_rank, num_heads_per_partition, num_kv_heads_per_partition, partition_head_l, partition_head_u = \
            get_tp_info(self, tp_size, tp_group, num_heads, num_kv_heads)

        # trick for calculating "interval sum of array with suffix 1s"
        num_dense_heads_for_partition = max(0, num_heads_per_partition +
                                            min(0, num_dense_heads - num_heads + partition_head_l))

        # if active_head_range:
        #     head_range_l, head_range_u = active_head_range
        #     head_range_l = max(head_range_l, partition_head_l)
        #     head_range_u = min(head_range_u, partition_head_u)
        #     active_head_range = (head_range_l, head_range_u)

        head_sliding_step = max(1, int(vert_stride / num_heads))
        head_sliding_offset += partition_head_l * head_sliding_step

        assert vert_stride >= 1, "Vertical stride should be position integer. Value 1 will collapse to dense attention."
        if vert_stride > 1:
            assert local_blocks >= 1, "Token in the first block will attend to nothing in some blocks."
        self.max_seq_len = multiple_of(max_seq_len,
                                       max([block_size, block_m or 1, block_n or 1, 64]))  # backward issue?
        self.num_heads = num_heads_per_partition
        self.num_kv_heads = num_kv_heads_per_partition
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.homo_head = homo_head
        self.num_dense_heads = num_dense_heads_for_partition
        print(f"[Layer {self.layer_number}][TPRank {tp_rank}] Before get_sparse_attn_mask")
        num_kv_heads_modified = None if homo_head else num_kv_heads
        print(f"[Layer {self.layer_number}][TPRank {tp_rank}] args: {num_heads}, {self.max_seq_len},"
              f"{block_size}, {local_blocks}, {vert_stride}, {homo_head}, {num_kv_heads_modified},"
              f"{head_sliding_offset}, {num_dense_heads}, {partition_head_l}, {partition_head_u} {num_kv_heads_per_partition}")
        sparse_pattern = get_sparse_attn_mask(num_heads, self.max_seq_len,
                                              block_size=block_size,
                                              local_blocks=local_blocks,
                                              vert_stride=vert_stride,
                                              homo_head=homo_head,
                                              dtype=torch.uint8,
                                              num_kv_heads=num_kv_heads_modified,
                                              head_sliding_offset=head_sliding_offset,
                                              num_dense_heads=num_dense_heads)[1]
        try:
            sparse_str = f"{sparse_pattern[0][::16, ::16].cpu().numpy().tolist()}".replace(", ", "").replace("][",
                                                                                                             "]\n [")
            print(
                f"[Layer {self.layer_number}][TPRank {tp_rank}] Sparse pattern[0]: (size) {sparse_pattern[0].shape} \\n\n{sparse_str}")
        except Exception:
            pass

        # no need to do this for homo heads, as patterns are the same across rangs
        if (not homo_head):
            if (active_head_range is None):
                active_head_range = (0, len(sparse_pattern))
            assert isinstance(active_head_range, tuple)
            assert len(active_head_range) == 2, '"active_head_range" should be a tuple of start/end index of the heads.'
            h_start, h_end = active_head_range
            h_start = max(h_start, partition_head_l)
            h_end = min(h_end, partition_head_u)
            sparse_pattern = sparse_pattern[h_start:h_end]
        print(
            f"[Layer {self.layer_number}][TPRank {tp_rank}] Before super: {block_size}, {sparse_pattern.shape if not isinstance(sparse_pattern, list) else len(sparse_pattern)},"
            f" {block_m}, {block_n}, {tp_size}, {kwargs}")
        super().__init__(block_size, sparse_pattern, block_m=block_m, block_n=block_n,
                         tp_size=tp_size, tp_group=tp_group, **kwargs)


class DKernelLocalStrideSparseAttention(torch.nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_number: int,
            attn_mask_type: AttnMaskType,
            attention_type: str,
            attention_dropout: float = None,
    ):
        super().__init__()
        self.config = config
        self.dkernel_forward_mask_type = False
        self.qkv_format: str = "bshd"
        self.seq_dim = 1  # bshd
        self.attn_mask_type = attn_mask_type.name
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout

        if self.config.apply_query_key_layer_scaling != bool(
                int(os.getenv("DKERNEL_APPLY_QUERY_KEY_LAYER_SCALING", 0))
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable DKERNEL_APPLY_QUERY_KEY_LAYER_SCALING is "
                f"{os.getenv('DKERNEL_APPLY_QUERY_KEY_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        extra_kwargs = {}

        # num_gqa_groups

        # attention_type

        # self.dkernel_forward_mask_type?

        # cp_{group, global_ranks, stream}
        assert (
                self.config.context_parallel_size == 1
        ), "DKernelLocalStrideSparseAttention does not support context parallelism."

        sparse_local_blocks = self.config.sparse_local_blocks

        # deterministic mode??

        # window size????
        if self.config.window_size is not None:
            assert self.config.window_size[1] == 0, "Only accepts causal window"
            assert self.config.window_size[
                       0] % self.config.sparse_block_size == 0, "DKernelLocalStrideSparseAttention requires window size to be divisible by block size."
            new_sparse_local_blocks = self.config.window_size[0] // self.config.sparse_block_size
            if sparse_local_blocks > new_sparse_local_blocks:
                warnings.warn(
                    "Window size is smaller than the accumulated size of local blocks of the default sparse attention pattern. In this case, sliding window is implemented by reducing the local blocks in the pattern.")
                warnings.warn(
                    f"The default local blocks is {sparse_local_blocks}, and the calculated value is {new_sparse_local_blocks} based on window size {self.config.window_size} and sparse_block_size {self.config.sparse_block_size}.")
                sparse_local_blocks = new_sparse_local_blocks

        vert_stride = self.config.sparse_vertical_stride
        head_sliding_offset = self.config.sparse_head_sliding_offset
        if vert_stride == 0:  # if vertical stride is not used, we need to modify it to fit into DKernel
            vert_stride = ((self.config.sparse_max_seq_len + self.config.sparse_block_size - 1)
                           // self.config.sparse_block_size) * self.config.num_attention_heads + 1
            head_sliding_offset = 1
            # this is to trick the DKernel to not generate any vertical strips
            warnings.warn("Modifying vert_stride and head_sliding_offset to not generate vertical strips"
                          " as vert_stride is set to 0.")

        tp_size = self.config.tensor_model_parallel_size
        tp_group = get_tensor_model_parallel_group(check_initialized=False)

        self.tp_size = tp_size
        self.tp_group = tp_group

        extra_kwargs["tp_size"] = tp_size
        extra_kwargs["tp_group"] = tp_group
        extra_kwargs["layer_number"] = layer_number

        self.layer_number = layer_number
        if self.config.sparse_homo_head:
            self.modified_num_kv_heads = None
        else:
            self.modified_num_kv_heads = self.config.num_query_groups
        print(f"LocalStrideSparseAttention({self.config.num_attention_heads}, {self.config.sparse_max_seq_len},"
              f" {self.config.sparse_block_size}, {sparse_local_blocks}, {vert_stride},"
              f" seq_dim={self.seq_dim}, homo_head={self.config.sparse_homo_head}, num_dense_heads={self.config.sparse_num_dense_heads},"
              f" num_kv_heads={self.modified_num_kv_heads}, active_head_range={self.config.sparse_active_head_range},"
              f" head_sliding_offset={head_sliding_offset}, block_m={self.config.sparse_block_m},"
              f" block_n={self.config.sparse_block_n}, extra_kwargs={extra_kwargs})")
        self.localStrideSparseAttention = LocalStrideSparseAttentionForTP(
            num_heads=self.config.num_attention_heads,
            max_seq_len=self.config.sparse_max_seq_len,  # ??
            block_size=self.config.sparse_block_size,
            local_blocks=sparse_local_blocks,
            vert_stride=vert_stride,
            # *,
            seq_dim=self.seq_dim,
            homo_head=self.config.sparse_homo_head,
            num_dense_heads=self.config.sparse_num_dense_heads,
            num_kv_heads=self.modified_num_kv_heads,
            active_head_range=self.config.sparse_active_head_range,
            head_sliding_offset=head_sliding_offset,
            block_m=self.config.sparse_block_m,
            block_n=self.config.sparse_block_n,
            **extra_kwargs  # for SparseAttention
        )
        # print()

        q, k, v = [torch.rand(1, 2, 8, 64, device="cuda").requires_grad_() for _ in range(3)]
        attn = LocalStrideSparseAttention(
            1, 128, 64, 32, vert_stride, seq_dim=1, homo_head=True,
            num_dense_heads=self.config.sparse_num_dense_heads,
            num_kv_heads=self.modified_num_kv_heads,
            active_head_range=self.config.sparse_active_head_range,
            head_sliding_offset=head_sliding_offset,
            block_m=self.config.sparse_block_m,
            block_n=self.config.sparse_block_n,
        )
        attn.to("cuda")
        print("Try localStrideSparseAttention")
        attn(q, k, v)
        # print("Try localStrideSparseAttention #2")
        # self.localStrideSparseAttention(q, k, v)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attention_mask: Tensor,
            attn_mask_type: AttnMaskType,
            packed_seq_params: PackedSeqParams = None,
            **kwargs
    ):
        packed_seq_params = (
            dataclasses.asdict(packed_seq_params) if packed_seq_params is not None else {}
        )

        if self.config.apply_rope_fusion:
            self.qkv_format = 'bshd'

        qkv_format = packed_seq_params.get("qkv_format", self.qkv_format)

        tp_info = get_tp_info(self, self.tp_size, self.tp_group, self.config.num_attention_heads,
                              self.config.num_query_groups)
        tp_rank, num_heads_per_partition, num_kv_heads_per_partition, partition_head_l, partition_head_u = tp_info
        assert qkv_format in ["bshd", "sbhd"], f"Unsupported qkv_format: {qkv_format}"
        # print(
        #     f"[Layer {self.layer_number} Rank {tp_rank}] Before transpose: {query.shape}, {key.shape}, {value.shape}, {qkv_format}, {self.qkv_format}")

        if self.config.apply_rope_fusion and qkv_format == 'bshd':  # input is sbhd, and we need to transpose it
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
            # In PyTorch, the following two tensors are in fact the same:
            #   Tensor with shape (1, S, H, D) and stride (S*H*D, H*D, D, 1)
            #   Tensor with shape (1, S, H, D) and stride (H*D, H*D, D, 1)
            # Stride for a dimension that is 1 has no meaning, so tensors created two different ways
            # can have same shape but different strides.
            # We unify them to the first one to pass the stride check in TE
            if value.shape == key.shape and value.shape[1] == 1 and value.stride() != key.stride():
                value = value.as_strided(value.shape, key.stride())

        query = query.reshape(query.shape[0], query.shape[1], self.localStrideSparseAttention.num_heads,
                              self.config.hidden_size // self.config.num_attention_heads)
        key = key.reshape(key.shape[0], key.shape[1], num_kv_heads_per_partition,
                          self.config.hidden_size // self.config.num_attention_heads)
        value = value.reshape(value.shape[0], value.shape[1], num_kv_heads_per_partition,
                              self.config.hidden_size // self.config.num_attention_heads)
        print(f"[Layer {self.layer_number} Rank {tp_rank}] key.shape", key.shape)
        # key = key.expand(-1, -1, -1, self.config.num_attention_heads // self.config.num_query_groups, -1)
        key = key.repeat_interleave(self.config.num_attention_heads // self.config.num_query_groups, dim=2)
        value = value.repeat_interleave(self.config.num_attention_heads // self.config.num_query_groups, dim=2)

        # query = query.transpose(2, 3)
        # query = query.reshape(query.shape[0], query.shape[1], self.config.num_attention_heads, -1)
        packed_seq_params = {
            "cu_seqlen_k": None,
            "cu_seqlen_q": None,
        }

        if self.attn_mask_type:
            if attn_mask_type == AttnMaskType.padding:
                attention_mask = attention_mask.to(torch.bool)
                # get left paddings
                left_paddings = attention_mask.argmax(dim=-1).view(-1)
                # get right paddings
                right_paddings = attention_mask.flip(dims=[-1]).argmax(dim=-1).view(-1)
                if left_paddings.sum() > 0:
                    packed_seq_params["left_paddings"] = left_paddings
                if right_paddings.sum() > 0:
                    packed_seq_params["seqlens"] = attention_mask.shape[-1] - right_paddings
        # print(f"[Layer {self.layer_number} Rank {tp_rank}] Before super forward: {query.shape}, {key.shape}, {value.shape}, {packed_seq_params}")
        core_attn_out = self.localStrideSparseAttention.forward(
            query,
            key,
            value,
            sm_scale=None,
            # *,
            **packed_seq_params
        )
        # print(f"[Layer {self.layer_number} Rank {tp_rank}] After super forward: {core_attn_out.shape}")
        if self.config.apply_rope_fusion and qkv_format == "bshd":
            core_attn_out = core_attn_out.transpose(0, 1)
        # core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], self.config.num_attention_heads // self.config.num_query_groups, self.config.num_query_groups, -1)
        # core_attn_out = core_attn_out.transpose(2, 3)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        # with open(f"/u/yufengd4/core_sparse_{tp_rank}.out", "w") as f:
        #     import json
        #     json.dump(core_attn_out.detach().cpu().tolist(), f)
        # print(core_attn_out)
        # input()
        # return shape [s, b, h*d]
        return core_attn_out


class DKernelPredefinedSparseAttention(torch.nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_number: int,
            attn_mask_type: AttnMaskType,
            attention_type: str,
            attention_dropout: float = None,
    ):
        super().__init__()
        self.config = config
        self.qkv_format: str = "bshd"
        self.seq_dim = 1  # bshd
        self.attn_mask_type = attn_mask_type.name
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout

        if self.config.apply_query_key_layer_scaling != bool(
                int(os.getenv("DKERNEL_APPLY_QUERY_KEY_LAYER_SCALING", 0))
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable DKERNEL_APPLY_QUERY_KEY_LAYER_SCALING is "
                f"{os.getenv('DKERNEL_APPLY_QUERY_KEY_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        extra_kwargs = {}

        # num_gqa_groups

        # cp_{group, global_ranks, stream}
        assert (
                self.config.context_parallel_size == 1
        ), "DKernelLocalStrideSparseAttention does not support context parallelism."

        sparse_local_blocks = self.config.sparse_local_blocks

        # block_size = 128

        tp_size = self.config.tensor_model_parallel_size
        tp_group = get_tensor_model_parallel_group(check_initialized=False)

        self.tp_size = tp_size
        self.tp_group = tp_group
        tp_rank = torch.distributed.get_rank(group=tp_group)

        extra_kwargs["tp_size"] = tp_size
        extra_kwargs["tp_group"] = tp_group
        extra_kwargs["layer_number"] = layer_number

        self.layer_number = layer_number

        print(f"DKernelSparseAttention({self.config.num_attention_heads}, {self.config.sparse_max_seq_len},"
              f" {self.config.sparse_block_size}, {sparse_local_blocks}, "
              f" seq_dim={self.seq_dim},"
              f" block_m={self.config.sparse_block_m},"
              f" block_n={self.config.sparse_block_n}, layer_number={layer_number}), tp_rank={tp_rank}")

        seqlen = self.config.sparse_max_seq_len  # 128k
        block_size = self.config.sparse_block_size
        blocks = (seqlen - 1) // block_size + 1  # ceil
        # split 128k context length into 8 blocks
        num_oracles = 32  # Magic number

        self.num_oracles = num_oracles
        oracle_size = seqlen // num_oracles
        self.oracle_size = oracle_size

        # force modify local blocks (sliding window)
        sparse_local_blocks = oracle_size // block_size
        self.sparse_local_blocks = oracle_size // block_size
        self.sparse_window_size = sparse_local_blocks * block_size  # avoid using "window_size" since it can be modified

        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        # patterns: 28 + 8 + 1
        self.patterns = []
        self.attentions = []
        with (torch.no_grad()):
            q_pos = torch.arange(blocks)[None, :, None]  # [1, b, 1]
            k_pos = torch.arange(blocks)[None, None]  # [1, 1, b]
            causal_mask = (q_pos >= k_pos)
            sliding_window_mask = (q_pos - k_pos < sparse_local_blocks)

            def create_mask(vert_mask_list):
                mask_vert = torch.zeros(blocks).bool()
                for col in vert_mask_list:
                    real_col = int(col * oracle_size + 0.5)
                    col_block_start = real_col // block_size
                    col_block_end = (real_col + oracle_size) // block_size
                    mask_vert[col_block_start:col_block_end] = True
                mask_vert[0:1] = True  # attention sink
                return (causal_mask & (sliding_window_mask | mask_vert[None, None, :])
                        ).to(device).to(dtype=torch.bfloat16)

            # attend to one
            for i in range(num_oracles):
                self.patterns.append(create_mask((i,)))

            for i in range(num_oracles - 1):
                self.patterns.append(create_mask((i + 0.5,)))
            # attend to two
            # for i in range(1, num_oracles):
            #     for j in range(i):
            #         self.patterns.append(create_mask((i, j)))

            # attend to all
            self.patterns.append(causal_mask.to(device).to(dtype=torch.bfloat16))

            # test mask
            # self.patterns.append(create_mask((2, 3, 4, 5)))

        for i, pattern in enumerate(self.patterns):
            if tp_rank == 0 and self.layer_number == 1:
                st = len(pattern[0]) // 64  # draw a 64x64 matrix
                sparse_str = f"{pattern[0][::st, ::st].to(dtype=torch.int).cpu().numpy().tolist()}".replace(", ",
                                                                                                            "").replace(
                    "][",
                    "]\n [")
                print(
                    f"[Layer {self.layer_number}][TPRank {tp_rank}] Sparse pattern[{i}]: (size) {pattern[0].shape} \\n\n{sparse_str}")
            self.attentions.append(SparseAttentionForTP(block_size, pattern,
                                                        seq_dim=self.seq_dim,
                                                        block_m=self.config.sparse_block_m,
                                                        block_n=self.config.sparse_block_n,
                                                        tp_size=tp_size, tp_group=tp_group))

        if tp_rank == 0 and self.layer_number == 1:
            q, k, v = [torch.rand(1, 8192, 8, 128, device="cuda").requires_grad_() for _ in range(3)]  # [b, s, h, d]
            print("Try PredefinedSparseAttention")
            import time
            start = time.time()
            o, p = self.attentions[-1](q, k, v, return_attn=True)
            print("First attention run: p.shape", p.shape, "; time:", time.time() - start)

    def get_block_one(self, start, end, primary_patterns_first=True):
        if primary_patterns_first:
            if start == end:
                return start
            else:
                return self.num_oracles + start
        # find the front-most block that contains end
        for i in range(self.num_oracles):
            if (i + 1) * self.oracle_size >= end:
                return i
            if i < self.num_oracles - 1 and (i + 1) * self.oracle_size + self.oracle_size // 2 >= end:
                return i + self.num_oracles

    def get_focused_positions_from_positions(self, positions):
        if positions is None:
            return None
        return FocusedPositions(self.config.sparse_block_size, positions, self.sparse_window_size)

    def get_focused_positions_from_oracle_positions(self, oracle_positions):
        focused_positions = self.get_focused_positions_from_positions(oracle_positions)
        if focused_positions:
            block_size = self.oracle_size
            if len(focused_positions.intervals) > 2:
                return None
            if len(focused_positions.intervals) > 1:
                rounded_intervals = [round_interval_by_block(interval, block_size)
                                     for interval in focused_positions.intervals]
                if rounded_intervals[0][0] == rounded_intervals[1][0]:
                    focused_positions.intervals = rounded_intervals[:1]
                else:
                    focused_positions.intervals = rounded_intervals
                return focused_positions

            if len(focused_positions.intervals) == 1:
                rounded_interval = round_interval_by_block(focused_positions.intervals[0], (block_size + 1) // 2)
                if rounded_interval[1] < block_size:
                    rounded_interval = (0, block_size)
                focused_positions.intervals = [[rounded_interval[1] - block_size, min(rounded_interval[1],
                                                                            self.config.sparse_max_seq_len)]]
                return focused_positions

    def get_focused_positions_from_pattern_id(self, pattern_id):
        if pattern_id < 0 or pattern_id == len(self.patterns) - 1:
            return FocusedPositions()  # None
        if pattern_id < self.num_oracles:
            return self.get_focused_positions_from_positions([[pattern_id * self.oracle_size,
                                                               (pattern_id + 1) * self.oracle_size]])
        if pattern_id < self.num_oracles * 2 - 1:
            pattern_id -= self.num_oracles
            return self.get_focused_positions_from_positions([[pattern_id * self.oracle_size + self.oracle_size // 2,
                                                               (pattern_id + 1) * self.oracle_size +
                                                               self.oracle_size // 2]])
        pattern_id -= self.num_oracles * 2 - 1
        # y = floor(sqrt(2z + 0.25) + 0.5)
        y = int((math.sqrt(2 * pattern_id + 0.25) - 0.5))
        x = pattern_id - y * (y - 1) // 2
        return self.get_focused_positions_from_positions([[self.oracle_size * x, self.oracle_size * (x + 1)],
                                                          [self.oracle_size * y, self.oracle_size * (y + 1)]])

    def choose_attention_id_from_extra_kwargs(self, extra_kwargs):
        if extra_kwargs.get("distance_between_positions", 0):
            # use full attention if there is distance between oracle positions
            return -1
        oracle_mode = extra_kwargs.get("oracle_mode", "off")
        if oracle_mode != "on":  # dynamic
            pattern_id = extra_kwargs.get("dynamic_pattern_id", None)
            if isinstance(pattern_id, list) and len(pattern_id) > 0:
                return pattern_id[0]
            else:
                return -1
        else:  # from oracle
            oracle_positions = extra_kwargs.get("oracle_positions", None)
            block_size = self.oracle_size
            if oracle_positions is None:
                return -1
            if len(oracle_positions) == 0:
                return -1
            focused_positions = self.get_focused_positions_from_oracle_positions(oracle_positions)
            if len(focused_positions.intervals) > 2:
                return -1
            if len(focused_positions.intervals) > 1:
                block_start_1, block_start_2 = (focused_positions.intervals[0][0] // block_size,
                                                focused_positions.intervals[1][0] // block_size)
                # z = x + y(y - 1) / 2, x < y
                # y = floor(sqrt(2z + 0.25) + 0.5)
                return self.num_oracles * 2 - 1 + block_start_1 + block_start_2 * (block_start_2 - 1) // 2
            if len(focused_positions.intervals) == 1:
                block_start = focused_positions.intervals[0][0]
                if block_start % block_size == 0 or block_start // block_size >= self.num_oracles - 1:
                    return block_start // block_size
                else:
                    return block_start // block_size + self.num_oracles

    def choose_attention_from_extra_kwargs(self, extra_kwargs):
        attention_id = self.choose_attention_id_from_extra_kwargs(extra_kwargs)
        if self.layer_number == 1:
            print(f"[rank {torch.distributed.get_rank()}] Attention id: {attention_id}", )
        return self.attentions[attention_id]

    def get_pattern_id_iter(self):
        for i in range(self.num_oracles):
            yield i
            if i < self.num_oracles - 1:
                yield i + self.num_oracles
        for i in range(self.num_oracles * 2 - 1, len(self.attentions) - 1):
            yield i
        yield len(self.attentions) - 1

    def extract_pattern_from_prob(self, q_len, kv_len, prob, granularity=1, head_reduction="mean", target_position=None,
                                  min_coverage=0.9):
        """
        prob: from kernel, [batch_size, heads_per_gpu, block_m, sequence_length]
        granularity: if > 1, group [sequence_length] into blocks, with block size `granularity`
        head_reduction: [number], "mean", "max", "majority". Reduce probs from different heads.
                     "majority" means after the pattern is extracted, use majority vote.
        target_position: the position in [block_m] to extract from prob.
        mean_coverage: The minimum probability sum to cover. If no patterns cover enough info, full attention will be used.
        """

        # extract the prob of first instance from row target position
        if target_position is None:
            target_position = (q_len - 1) % prob.shape[2]
        prob_target = prob[0, :, target_position, :kv_len]  # [heads_per_gpu, sequence_length]
        heads_per_gpu, sequence_length = prob_target.shape[0], prob_target.shape[1]
        # split sequence into blocks and calculate max values

        prob_target_bulk = prob_target[:, :sequence_length // granularity * granularity]  # [h, rounded_s]
        prob_target_remainder = prob_target[:, sequence_length // granularity * granularity:]  # [h, s%block]
        prob_target_bulk = prob_target_bulk.reshape(heads_per_gpu, sequence_length // granularity,
                                                    granularity)  # [h, s // b, b]

        prob_target_bulk = prob_target_bulk.max(dim=-1).values  # [h, s // b]
        if prob_target_remainder.shape[-1] > 0:
            prob_target_remainder = prob_target_remainder.reshape(heads_per_gpu, 1,
                                                                  prob_target_remainder.shape[1])  # [h, 1, r]
            prob_target_remainder = prob_target_remainder.max(dim=-1).values  # [h, 1]
        # else: [h, 0]

        prob_target = torch.cat((prob_target_bulk, prob_target_remainder), dim=1)  # [h, ceil(s/b)]

        # reduce heads
        pattern_reduction_dim = None
        if isinstance(head_reduction, int):
            prob_target = prob_target[head_reduction:head_reduction + 1, :]
        elif head_reduction == "mean":
            prob_target = prob_target.mean(dim=0, keepdim=True)
        elif head_reduction == "max":
            prob_target = prob_target.max(dim=0, keepdim=True).values
        elif head_reduction == "none":
            pass
        elif head_reduction == "majority":
            pattern_reduction_dim = 1
        else:
            raise ValueError(f"Wrong value: prob_target `{prob_target}`")

        # [1, ceil(s/b)]  or [h, ceil(s/b)] for majority

        best_pattern_id, max_coverage = -1, min_coverage * prob_target.sum(dim=pattern_reduction_dim)
        real_best_pattern_id, real_max_coverage = -1, 0
        for j in self.get_pattern_id_iter():
            focused_positions = self.get_focused_positions_from_pattern_id(j)
            if len(focused_positions.intervals) == 0:
                break  # skip the full attention
            all_intervals = focused_positions.get_all_positions(kv_len, attention_sink=True, granularity=granularity)

            if pattern_reduction_dim:
                sum_prob = torch.zeros_like(prob_target[:, 0])
            else:
                sum_prob = 0

            for start, end in all_intervals:
                sum_prob += prob_target[:, start:end].sum(dim=pattern_reduction_dim)

            if pattern_reduction_dim:
                if best_pattern_id == -1:
                    best_pattern_id = torch.zeros_like(sum_prob) - 1
                best_pattern_id[sum_prob > max_coverage] = j
                # no update anymore
                max_coverage[sum_prob > max_coverage] = 100 * prob_target.sum(dim=pattern_reduction_dim)
            else:
                if sum_prob > max_coverage:
                    max_coverage = 100 * prob_target.sum(dim=pattern_reduction_dim)  # no update anymore
                    best_pattern_id = j
                if sum_prob > real_max_coverage:
                    real_max_coverage = sum_prob
                    real_best_pattern_id = j
            # if torch.distributed.get_rank() == 0:
            #     print("Pattern", j, ": All intervals", all_intervals, "coverage", sum_prob, "max_coverage:", max_coverage)
        if head_reduction == "majority":
            votes = {}
            for pid in best_pattern_id:
                votes[pid.item()] = votes.get(pid.item(), 0) + 1
            best_pattern_id, most_votes = -1, 0
            for pid in votes:
                if votes[pid] > most_votes:
                    best_pattern_id, most_votes = pid, votes[pid]

        return best_pattern_id, max_coverage, prob_target, real_max_coverage / prob_target.sum(), real_best_pattern_id

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attention_mask: Tensor,
            attn_mask_type: AttnMaskType,
            packed_seq_params: PackedSeqParams = None,
            extra_kwargs: Optional[dict] = None,
            **kwargs
    ):
        # print(f"[rank {torch.distributed.get_rank}] DKernelPredefinedSparseAttention.forward")
        packed_seq_params = (
            dataclasses.asdict(packed_seq_params) if packed_seq_params is not None else {}
        )

        if self.config.apply_rope_fusion:
            self.qkv_format = 'bshd'

        qkv_format = packed_seq_params.get("qkv_format", self.qkv_format)

        tp_rank, num_heads_per_partition, num_kv_heads_per_partition, partition_head_l, partition_head_u = (
            get_tp_info(self, self.tp_size, self.tp_group, self.config.num_attention_heads,
                        self.config.num_query_groups))
        assert qkv_format in ["bshd", "sbhd"], f"Unsupported qkv_format: {qkv_format}"

        if self.config.apply_rope_fusion and qkv_format == 'bshd':  # input is sbhd, and we need to transpose it
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
            # In PyTorch, the following two tensors are in fact the same:
            #   Tensor with shape (1, S, H, D) and stride (S*H*D, H*D, D, 1)
            #   Tensor with shape (1, S, H, D) and stride (H*D, H*D, D, 1)
            # Stride for a dimension that is 1 has no meaning, so tensors created two different ways
            # can have same shape but different strides.
            # We unify them to the first one to pass the stride check in TE
            if value.shape == key.shape and value.shape[1] == 1 and value.stride() != key.stride():
                value = value.as_strided(value.shape, key.stride())

        query = query.reshape(query.shape[0], query.shape[1], num_heads_per_partition,
                              self.config.hidden_size // self.config.num_attention_heads)
        key = key.reshape(key.shape[0], key.shape[1], num_kv_heads_per_partition,
                          self.config.hidden_size // self.config.num_attention_heads)
        value = value.reshape(value.shape[0], value.shape[1], num_kv_heads_per_partition,
                              self.config.hidden_size // self.config.num_attention_heads)
        # key = key.expand(-1, -1, -1, self.config.num_attention_heads // self.config.num_query_groups, -1)
        key = key.repeat_interleave(self.config.num_attention_heads // self.config.num_query_groups, dim=2)
        value = value.repeat_interleave(self.config.num_attention_heads // self.config.num_query_groups, dim=2)

        q_len = query.shape[self.seq_dim]  # len of updated tokens
        kv_len = key.shape[self.seq_dim]  # len of total tokens

        packed_seq_params = {
            "cu_seqlen_k": None,
            "cu_seqlen_q": None,
        }

        if self.attn_mask_type:
            if attn_mask_type == AttnMaskType.padding:
                attention_mask = attention_mask.to(torch.bool)
                # get left paddings
                left_paddings = attention_mask.argmax(dim=-1).view(-1)
                # get right paddings
                right_paddings = attention_mask.flip(dims=[-1]).argmax(dim=-1).view(-1)
                if left_paddings.sum() > 0:
                    packed_seq_params["left_paddings"] = left_paddings
                if right_paddings.sum() > 0:
                    packed_seq_params["seqlens"] = attention_mask.shape[-1] - right_paddings

        attention = self.choose_attention_from_extra_kwargs(extra_kwargs)

        core_attn_out, prob = attention.forward(
            query,
            key,
            value,
            sm_scale=None,
            return_attn=True,
            # *,
            **packed_seq_params
        )

        rounded_kv_len = 16384  # the smallest visual len from 16384
        while rounded_kv_len < kv_len:
            rounded_kv_len *= 2
        visual_block_count = 64
        visual_block_size = rounded_kv_len // visual_block_count

        # compare prob with pattern
        if self.layer_number == 1 and len(extra_kwargs["dynamic_pattern_id"]) == 0:  # at layer 2
            extracted_pattern_id_global, _, prob_target, _, _ = self.extract_pattern_from_prob(
                q_len, kv_len, prob, granularity=visual_block_size, head_reduction="none")
            values_int_tensor = [-1, -1, -1, -1]
            values_int_tensor[torch.distributed.get_rank()] = extracted_pattern_id_global
            for i in range(4):
                xi = broadcast_int_list(1, int_list=values_int_tensor[i:i+1], rank=i)
                values_int_tensor[i] = xi[0].item()
            scores = {}
            for i in range(4):
                scores[values_int_tensor[i]] = scores.get(values_int_tensor[i], 0) + 1
            extracted_pattern_id_global = -1
            for pattern in scores:
                if scores[pattern] >= 2:
                    extracted_pattern_id_global = pattern
                    break

            def get_visual_prob(contents_to_print, pattern_id):
                # calculate the maximum prob for each head
                max_prob_for_each_head = contents_to_print.max(dim=-1, keepdim=True).values + 1e-8
                # normalize the prob for each head
                contents_to_print = contents_to_print / max_prob_for_each_head
                # use ansi escape code from red to green to map 0 - 1
                red_values = (255 * (1 - contents_to_print)).to(int)
                green_values = (255 * contents_to_print).to(int)
                probs_to_digits = (10 * contents_to_print).to(int)
                probs_to_digits[probs_to_digits == 10] = 9
                probs_to_digits2 = probs_to_digits.tolist()

                # deal with pattern
                focused_positions = self.get_focused_positions_from_pattern_id(pattern_id)

                sw, bs = focused_positions.window_size, focused_positions.block_size
                if len(focused_positions.intervals) == 0 or sw == -1:
                    b1s_block, b1e_block, sws_block, swe_block = -1, -1, -1, -1

                else:
                    b1s, b1e = focused_positions.intervals[0]
                    b1s_block = b1s // visual_block_size
                    b1e_block = (b1e - 1) // visual_block_size + 1
                    sliding_window_start = focused_positions.get_actual_window(kv_len)[1]
                    sliding_window_end = kv_len
                    sws_block = sliding_window_start // visual_block_size
                    swe_block = (sliding_window_end - 1) // visual_block_size + 1
                    # ignore b2
                for j in range(len(probs_to_digits2)):
                    if b1s_block <= j < b1e_block or sws_block <= j < swe_block:
                        probs_to_digits2[j] = f"\033[5;4;48;2;{red_values[j].item()};{green_values[j].item()};0m{probs_to_digits2[j]}\033[0m"
                    else:
                        probs_to_digits2[j] = f"\033[48;2;{red_values[j].item()};{green_values[j].item()};0m{probs_to_digits2[j]}\033[0m"

                visual_prob = ''.join(probs_to_digits2)
                return visual_prob
            # visual_prob = get_visual_prob(prob_target[0], extracted_pattern_id_global)
            # print_string = f"[Rank {tp_rank}] {visual_prob} {extracted_pattern_id_global}"
            # print(print_string)

            if isinstance(extra_kwargs.get("dynamic_pattern_id", None), list):
                if len(extra_kwargs["dynamic_pattern_id"]) == 0:
                    extra_kwargs["dynamic_pattern_id"].append(extracted_pattern_id_global)
                elif extra_kwargs["dynamic_pattern_id"][0] == -1:
                    extra_kwargs["dynamic_pattern_id"][0] = extracted_pattern_id_global

        if self.config.apply_rope_fusion and qkv_format == "bshd":
            core_attn_out = core_attn_out.transpose(0, 1)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        # return shape [s, b, h*d]
        return core_attn_out
