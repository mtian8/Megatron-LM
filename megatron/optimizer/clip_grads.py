# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradient clipping."""

import torch
from torch._six import inf

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import mpu
from megatron.model.module import param_is_not_shared
from megatron.mpu.layers import param_is_not_tensor_parallel_duplicate


# >>>
from lutil import pax, tp
DEBUG_ITERATION = 1
# <<<

def clip_grad_norm_fp32(parameters, max_norm, norm_type=2, ITERATION=None):
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    # >>>
    raise Exception("currently debugging ... don't call me.")
    # <<<

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    grads = []
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none:
            grad = param.grad.detach()
        if grad_not_none:
            # Make sure the grads are in fp32
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            grads.append(grad)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grads_for_norm.append(grad)
        # >>>
        # else:
        #     pax(1, {
        #         "grad_not_none" : grad_not_none,
        #         "is_not_shared" : is_not_shared,
        #         "is_not_tp_duplicate" : is_not_tp_duplicate,
        #     })
        # <<<

    # >>>
    # if ITERATION == DEBUG_ITERATION:
    #     pax(0, {
    #         "[LOC]" : "[** BEFORE CALC NORM **]",
    #         "[ITERATION]" : ITERATION,
    #         "max_norm" : max_norm,
    #         "parameters" : parameters,
    #         # "grads" : grads,
    #         "grads_for_norm" : grads_for_norm,
    #     })
    # <<<

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            # >>>
            # pax(1, {
            #     # "fn" : amp_C.multi_tensor_l2norm,
            #     "dummy_overflow_buf" : tp(dummy_overflow_buf),
            #     "grads_for_norm" : grads_for_norm,
            # })
            # <<<
            grad_norm, _ = multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [grads_for_norm],
                False # no per-parameter norm
            )
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # >>>
        # if ITERATION == DEBUG_ITERATION:
        #     pax(0, {
        #         "[LOC]" : "[** CALC NORM **]",
        #         "[ITERATION]" : ITERATION,
        #         "max_norm" : max_norm,
        #         "norm_type" : norm_type,
        #         "grad_norm" : tp(grad_norm),
        #         "total_norm" : tp(total_norm),
        #     })
        # <<<

        # Sum across all model-parallel GPUs.
        # >>>
        from megatron import get_args
        args = get_args()
        if not args.use_distributed_optimizer:
            torch.distributed.all_reduce(total_norm,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=mpu.get_model_parallel_group())
        # +++
        else:
            torch.distributed.all_reduce(total_norm,
                                         op=torch.distributed.ReduceOp.SUM)
        # <<<
        total_norm = total_norm.item() ** (1.0 / norm_type)

        # >>>
        # if ITERATION == DEBUG_ITERATION:
        #     pax(0, {
        #         "[LOC]" : "[** AFTER REDUCE. **]",
        #         "[ITERATION]" : ITERATION,
        #         "max_norm" : max_norm,
        #         "norm_type" : norm_type,
        #         "grad_norm" : grad_norm.item(),
        #         "total_norm" : total_norm,
        #     })
        # <<<

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        dummy_overflow_buf = torch.cuda.IntTensor([0])
        multi_tensor_applier(amp_C.multi_tensor_scale,
                             dummy_overflow_buf,
                             [grads, grads],
                             clip_coeff)

    # >>>
    # # from pygit2 import Repository
    # if ITERATION == DEBUG_ITERATION:
    #     pax(1, {
    #         "[LOC]" : "[** CLIP / FINAL **]",
    #         "[ITERATION]" : ITERATION,
    #         "grads" : grads,
    #         "clip_coeff" : tp(clip_coeff),
    #         # "repo" : Repository('.').head.shorthand,
    #     })
    # <<<

    return total_norm


def count_zeros_fp32(parameters):

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = 0.0
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(total_num_zeros,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    total_num_zeros = total_num_zeros.item()

    return total_num_zeros
