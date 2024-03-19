from dataclasses import dataclass


# TODO : Have an update class that can add more key value pairs
@dataclass
class CommonInferenceParams:
    use_greedy: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    return_log_probs: bool = False
    num_tokens_to_generate:int = 30
