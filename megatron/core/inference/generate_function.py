from typing import List, Tuple, Union

from torch import Tensor
import torch
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.core.inference.backends.abstract_backend import AbstractBackend
from megatron.core.inference.backends.mcore_backend import MCoreBackend
from megatron.core.inference.backends.trt_llm_backend import TRTLLMBackend
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core import mpu

def common_generate(inference_backend: Union[MCoreBackend, TRTLLMBackend], prompts:List[str] = None, common_inference_params: CommonInferenceParams = None) -> Tuple[Tensor, List[str], Tensor]:
    """Common Generate function to call for inference

    This function will automatically chose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. 

    Args:
        inference_backend (Union[MCoreBackend, TRTLLMBackend]): The inference backend, that has the generate function.
        prompts (List[str], optional): The input prompts as a list of strings. Typically of length global batch size. Defaults to None.
        common_inference_params (CommonInferenceParams, optional): The usual inference parameters that are used for generation. Defaults to None.

    Returns:
        Tuple[Tensor, List[str], Tensor]: A tuple of all the generated tokens , all the generated texts and optionally the output log probabilities of the token 
    """   
    prompts_tokens_with_generations, prompts_plus_generations_detokenized, output_log_probs  = inference_backend.generate(prompts=prompts, common_inference_params=common_inference_params)

    return prompts_tokens_with_generations, prompts_plus_generations_detokenized, output_log_probs 



 