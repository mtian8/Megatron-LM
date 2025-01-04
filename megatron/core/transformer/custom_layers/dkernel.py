import dataclasses
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

import dkernel
from dkernel import SparseAttention, LocalStrideSparseAttention
from dkernel.utils import multiple_of, get_sparse_attn_mask

import warnings

def get_dkernel_version():
    def get_dkernel_str():
        if hasattr(dkernel, '__version__'):
            return str(dkernel.__version__)
        else:
            return version("dkernel")
    return (packaging.version.Version(get_dkernel_str()))

_dkernel_version = get_dkernel_version()


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
                 seq_dim: Optional[int]=None,
                 block_m: Optional[int]=None,
                 block_n: Optional[int]=None,
                 tp_size: int=1,
                 tp_group: Optional[torch.distributed.ProcessGroup]=None,
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
    print(f"[Layer {self.layer_number}][TPRank {tp_rank}] In LocalStride: tp_size={tp_size},"
          f" num_heads={num_heads}, num_kv_heads={num_kv_heads}")

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
                 homo_head: bool=False,
                 num_dense_heads: int=0,
                 num_kv_heads: Optional[int]=None,
                 active_head_range: Optional[Tuple[int, int]]=None,
                 head_sliding_offset: int=0,
                 block_m: Optional[int]=None,
                 block_n: Optional[int]=None,
                 tp_size: int=1,
                 tp_group: Optional[torch.distributed.ProcessGroup]=None,
                 layer_number: int=None,
                 **kwargs
                 ):

        # tensor parallelism
        self.layer_number = layer_number
        tp_rank, num_heads_per_partition, num_kv_heads_per_partition, partition_head_l, partition_head_u =\
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
        self.max_seq_len = multiple_of(max_seq_len, max([block_size, block_m or 1, block_n or 1, 64]))  # backward issue?
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
              f"{head_sliding_offset}, {num_dense_heads}, {partition_head_l}, {partition_head_u}")
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
            print(f"[Layer {self.layer_number}][TPRank {tp_rank}] Sparse pattern[0]: \\n\n{sparse_pattern[0][:32][:32].cpu().numpy()}")
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
        print(f"[Layer {self.layer_number}][TPRank {tp_rank}] Before super: {block_size}, {sparse_pattern.shape if not isinstance(sparse_pattern, list) else len(sparse_pattern)},"
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
            assert self.config.window_size[0] % self.config.sparse_block_size == 0, "DKernelLocalStrideSparseAttention requires window size to be divisible by block size."
            new_sparse_local_blocks = self.config.window_size[0] // self.config.sparse_block_size
            if sparse_local_blocks > new_sparse_local_blocks:
                warnings.warn("Window size is smaller than the accumulated size of local blocks of the default sparse attention pattern. In this case, sliding window is implemented by reducing the local blocks in the pattern.")
                warnings.warn(f"The default local blocks is {sparse_local_blocks}, and the calculated value is {new_sparse_local_blocks} based on window size {self.config.window_size} and sparse_block_size {self.config.sparse_block_size}.")
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
            max_seq_len=self.config.sparse_max_seq_len, # ??
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

        # q, k, v = [torch.rand(2, 8192, 32, 128, device="cuda").requires_grad_() for _ in range(3)]
        # attn = LocalStrideSparseAttention(
        #     32, 8192, 64, 32, vert_stride, seq_dim=1, homo_head=True,
        #     num_dense_heads=self.config.sparse_num_dense_heads,
        #     num_kv_heads=self.modified_num_kv_heads,
        #     active_head_range=self.config.sparse_active_head_range,
        #     head_sliding_offset=head_sliding_offset,
        #     block_m=self.config.sparse_block_m,
        #     block_n=self.config.sparse_block_n,
        # )
        # attn.to("cuda")
        # print("Try localStrideSparseAttention")
        # attn(q, k, v)
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
    ):
        packed_seq_params = (
            dataclasses.asdict(packed_seq_params) if packed_seq_params is not None else {}
        )

        if self.config.apply_rope_fusion:
            self.qkv_format = 'bshd'

        qkv_format = packed_seq_params.get("qkv_format", self.qkv_format)

        tp_info = get_tp_info(self, self.tp_size, self.tp_group, self.config.num_attention_heads, self.config.num_query_groups)
        tp_rank = tp_info[0]
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

        query = query.reshape(query.shape[0], query.shape[1], self.localStrideSparseAttention.num_heads, self.config.hidden_size // self.config.num_attention_heads)
        key = key.reshape(key.shape[0], key.shape[1], self.localStrideSparseAttention.num_kv_heads, self.config.hidden_size // self.config.num_attention_heads)
        value = value.reshape(value.shape[0], value.shape[1], self.localStrideSparseAttention.num_kv_heads, self.config.hidden_size // self.config.num_attention_heads)
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

