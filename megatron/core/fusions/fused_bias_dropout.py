# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import torch

from megatron.core.jit import jit_fuser


def _bias_dropout_add_func(x_with_bias, residual, prob, training):
    # print("In fused_bias_dropout.py, _bias_dropout_add_func()")
    # type: (Tuple[Tensor, Optional[Tensor]], Tensor, float, bool) -> Tensor
    # NOTE: Previously, the argument `bias` used to be passed as
    # `bias.expand_as(residual)` when the `bias_dropout_func` is called from the
    # transformer layer but broadcasting should automatically take care of that.
    # Also, looking at broadcasting semantics, `expand_as` and broadcasting
    # seem to be identical performance-wise (both just change the view).

    x, bias = x_with_bias  # unpack

    # If we want to train mixed precision, then the output of this function
    # should be half precision. However, in AMP O1, the input (residual) is
    # in fp32, and it will up-cast the result to fp32, causing pipeline parallel
    # GPU communication to hang. Therefore, we need to cast residual to the same
    # dtype as x.
    # print("x.dtype: ", x.dtype)
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)

    # The Dropout operation, Residual Addition and the tensor returning can be
    # done generically outside the if statement, but that stops fusing of Bias
    # Addition-Dropout-Residual Addition operation. So doing it together inside
    # the conditional branch to improve performance
    # print("bias: ", bias)
    if bias is not None:
        x = x + bias
        # print("before dropout")
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        # print("after dropout")
        out = residual + out
        # print("returning out")
        return out
    else:
        # print("before dropout (no bias): ", x.shape, prob, training)


        # print("Torch version: ", torch.__version__, torch.__file__) 
        # print('device count', torch.cuda.device_count())
        # print('current device', torch.cuda.current_device())
        # print('device name', torch.cuda.get_device_name())
        print('device properties', torch.cuda.get_device_properties(0))
        # a1 = torch.rand(1, dtype=torch.bfloat16, device="cuda:0")
        # print("Out_1")
        # out_a1 = torch.nn.functional.dropout(a1, p=prob, training=training)
        # print("Out_")
        
        # print("Out a1", out_a1[0])
        # a2 = torch.rand_like(x)
        # out_a2 = torch.nn.functional.dropout(a2, p=prob, training=training)
        # print("Out a2", out_a2[0][0]) 
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        # print("after dropout (no bias)")
        out = residual + out
        # print("returning out")
        return out


def bias_dropout_add_unfused(training):
    def _bias_dropout_add(x_with_bias, residual, prob):
        return _bias_dropout_add_func(x_with_bias, residual, prob, training)

    return _bias_dropout_add


@jit_fuser
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, True)


@jit_fuser
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, False)


def get_bias_dropout_add(training, fused):
    if fused:
        # jit scripting for a nn.module (with dropout) is not
        # triggering the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if training:
            return bias_dropout_add_fused_train
        else:
            return bias_dropout_add_fused_inference
    else:
        return bias_dropout_add_unfused(training)
