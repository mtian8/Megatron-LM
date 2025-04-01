# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


from megatron.core.models.gpt import GPTModel

class Llama3Model(GPTModel):
    """Llama Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        if "rope_scaling" not in kwargs:
            kwargs["rope_scaling"] = {
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        super().__init__(*args, **kwargs)