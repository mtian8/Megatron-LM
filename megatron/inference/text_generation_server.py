# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import datetime
import torch
import json
import threading
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api
from megatron.training import get_args
from megatron.inference.text_generation import generate_and_post_process
from megatron.inference.text_generation import beam_search_and_post_process
from megatron.inference.text_generation.tokenization import tokenize_prompts, detokenize_generations, tokenize_prompts_on_one_rank
from transformer_engine.pytorch.attention import check_set_window_size

GENERATE_NUM = 0
BEAM_NUM = 1
TOKENIZE_NUM = 2
DETOKENIZE_NUM = 3
MODIFY_WINDOW_SIZE_NUM = 4
GENERAL_API_NUM = 5
lock = threading.Lock()

class MegatronGenerate(Resource):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def send_do_generate():
        choice = torch.tensor(GENERATE_NUM, dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
     
    @staticmethod
    def send_do_beam_search():
        choice = torch.tensor(BEAM_NUM, dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
    
    def put(self):
        return self.put_inner(request.get_json())

    def put_inner(self, request_json):
        args = get_args()
       
        if not "prompts" in request_json:
            return "prompts argument required", 400
        
        if "max_len" in request_json:
            return "max_len is no longer used.  Replace with tokens_to_generate", 400
        
        if "sentences" in request_json:
            return "sentences is no longer used.  Replace with prompts", 400

        prompts = request_json["prompts"]
        if not isinstance(prompts, list):
            return "prompts is not a list of strings", 400

        if len(prompts) == 0:
            return "prompts is empty", 400
        
        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400
        
        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request_json:
            tokens_to_generate = request_json["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 0:
                return "tokens_to_generate must be an integer greater than or equal to 0"

        logprobs = False
        if "logprobs" in request_json:
            logprobs = request_json["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"
        
        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"
        
        temperature = 1.0
        if "temperature" in request_json:
            temperature = request_json["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"
        
        top_k = 0.0
        if "top_k" in request_json:
            top_k = request_json["top_k"]
            if not (type(top_k) == int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"
        
        top_p = 0.0
        if "top_p" in request_json:
            top_p = request_json["top_p"]
            if not (type(top_p) == float):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"
        
        top_p_decay = 0.0
        if "top_p_decay" in request_json:
            top_p_decay = request_json["top_p_decay"]
            if not (type(top_p_decay) == float):
                return "top_p_decay must be a positive float less than or equal to 1.0"
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p"
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0"
        
        top_p_bound = 0.0
        if "top_p_bound" in request_json:
            top_p_bound = request_json["top_p_bound"]
            if not (type(top_p_bound) == float):
                return "top_p_bound must be a positive float less than or equal to top_p"
            if top_p == 0.0:
                return "top_p_bound cannot be set without top_p"
            if not (0.0 < top_p_bound <= top_p):
                return "top_p_bound must be greater than 0 and less than top_p"
        
        add_BOS = False
        if "add_BOS" in request_json:
            add_BOS = request_json["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"
        
        # if any([len(prompt) == 0 for prompt in prompts]) and not add_BOS:
        #     return "Empty prompts require add_BOS=true"

        stop_on_double_eol = False
        if "stop_on_double_eol" in request_json:
            stop_on_double_eol = request_json["stop_on_double_eol"]
            if not isinstance(stop_on_double_eol, bool):
                return "stop_on_double_eol must be a boolean value"
        
        stop_on_eol = False
        if "stop_on_eol" in request_json:
            stop_on_eol = request_json["stop_on_eol"]
            if not isinstance(stop_on_eol, bool):
                return "stop_on_eol must be a boolean value"

        prevent_newline_after_colon = False
        if "prevent_newline_after_colon" in request_json:
            prevent_newline_after_colon = request_json["prevent_newline_after_colon"]
            if not isinstance(prevent_newline_after_colon, bool):
                return "prevent_newline_after_colon must be a boolean value"

        random_seed = -1
        if "random_seed" in request_json:
            random_seed = request_json["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0: 
                return "random_seed must be a positive integer"

        no_log = False
        if "no_log" in request_json:
            no_log = request_json["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"
        
        beam_width = None
        if "beam_width" in request_json:
            beam_width = request_json["beam_width"]
            if not isinstance(beam_width, int):
                return "beam_width must be integer"
            if beam_width < 1:
                return "beam_width must be an integer > 1"
            if len(prompts) > 1:
                return "When doing beam_search, batch size must be 1"

        stop_token=50256
        if "stop_token" in request_json:
            stop_token = request_json["stop_token"]
            if not isinstance(stop_token, int):
                return "stop_token must be an integer"
        
        length_penalty = 1 
        if "length_penalty" in request_json:
            length_penalty = request_json["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float"

        ignore_special_tokens = True
        if "ignore_special_tokens" in request_json:
            ignore_special_tokens = request_json["ignore_special_tokens"]
            if not isinstance(ignore_special_tokens, bool):
                return "ignore_special_tokens must be a boolean value"
        
        with lock:  # Need to get lock to keep multiple threads from hitting code
            
            if not no_log:
                print("request IP: " + str(request.remote_addr))
                print("All args:", flush=True)
                for k, v in request_json.items():
                    if k == "prompts":
                        print(f"{k}: {[v0[:10] + '...' for v0 in v]} ", flush=True)
                    else:
                        print(f"{k}: {v}", flush=True)
                # print(json.dumps(request_json),flush=True)
                print("start time: ", datetime.datetime.now())
            
            try:
                if beam_width is not None:
                    self.send_do_beam_search()  # Tell other ranks we're doing beam_search
                    response, response_seg, response_scores = \
                        beam_search_and_post_process(
                            self.model,
                            prompts=prompts,
                            tokens_to_generate=tokens_to_generate,
                            beam_size = beam_width,
                            add_BOS=add_BOS,
                            stop_token=stop_token,
                            num_return_gen=beam_width,  # Returning whole beam
                            length_penalty=length_penalty,
                            prevent_newline_after_colon=prevent_newline_after_colon,
                            ignore_special_tokens=ignore_special_tokens
                        )
                    
                    return jsonify({"text": response,
                        "segments": response_seg,
                        "scores": response_scores})
                else:
                    self.send_do_generate()  # Tell other ranks we're doing generate
                    response, response_seg, response_logprobs, _ = \
                        generate_and_post_process(
                            self.model,
                            prompts=prompts,
                            tokens_to_generate=tokens_to_generate,
                            return_output_log_probs=logprobs,
                            top_k_sampling=top_k,
                            top_p_sampling=top_p,
                            top_p_decay=top_p_decay,
                            top_p_bound=top_p_bound,
                            temperature=temperature,
                            add_BOS=add_BOS,
                            use_eod_token_for_early_termination=True,
                            stop_on_double_eol=stop_on_double_eol,
                            stop_on_eol=stop_on_eol,
                            prevent_newline_after_colon=prevent_newline_after_colon,
                            random_seed=random_seed,
                            ignore_special_tokens=ignore_special_tokens
                        )

                    return jsonify({"text": response,
                        "segments": response_seg,
                        "logprobs": response_logprobs})

            except ValueError as ve:
                return ve.args[0], 400
            print("end time: ", datetime.datetime.now())


class MegatronTokenize(Resource):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def send_do_tokenize():
        choice = torch.tensor([TOKENIZE_NUM], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)

    @staticmethod
    def send_do_detokenize():
        choice = torch.tensor([DETOKENIZE_NUM], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)

    def put(self):
        args = get_args()

        if not "texts" in request.get_json():
            return "texts argument required", 400

        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        texts = request.get_json()["texts"]
        if not isinstance(texts, list):
            return "texts is not a list of strings", 400

        if len(texts) == 0:
            return "texts is empty", 400

        if len(texts) > 128:
            return "Maximum number of texts is 128", 400

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        # if any([len(prompt) == 0 for prompt in texts]) and not add_BOS:
        #     return "Empty prompts require add_BOS=true"


        no_log = False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"

        ignore_special_tokens = False
        if "ignore_special_tokens" in request.get_json():
            ignore_special_tokens = request.get_json()["ignore_special_tokens"]
            if not isinstance(ignore_special_tokens, bool):
                return "ignore_special_tokens must be a boolean value"


        with lock:  # Need to get lock to keep multiple threads from hitting code
            # self.send_do_tokenize()
            if not no_log:
                print("request IP: " + str(request.remote_addr))
                # print(json.dumps(request.get_json()), flush=True)
                print("start time: ", datetime.datetime.now())

            try:
                prompts_tokens_tensor, _ = tokenize_prompts_on_one_rank(
                    prompts=texts,
                    tokens_to_generate=0,
                    add_BOS=add_BOS,
                    ignore_special_tokens=ignore_special_tokens
                )
                print("After tokenize_prompts", flush=True)
                cpu_tensor = prompts_tokens_tensor.cpu()
                # print("After cpu_tensor", flush=True)
                ret = cpu_tensor.tolist()
                # print("After tolist", flush=True)
                return jsonify({"token_ids": ret})

            except ValueError as ve:
                return ve.args[0]
            print("end time: ", datetime.datetime.now())


class MegatronDetokenize(Resource):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def send_do_detokenize():
        choice = torch.tensor([DETOKENIZE_NUM], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)

    def put(self):

        args = get_args()

        if not "tokens" in request.get_json():
            return "tokens argument required", 400

        tokens = request.get_json()["tokens"]
        if not isinstance(tokens, list):
            return "tokens is not a list of integer lists", 400

        if len(tokens) == 0:
            return "tokens is empty", 400

        if len(tokens) > 128:
            return "Maximum number of tokens is 128", 400

        no_log = not False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"

        ignore_special_tokens = True
        if "ignore_special_tokens" in request.get_json():
            ignore_special_tokens = request.get_json()["ignore_special_tokens"]
            if not isinstance(ignore_special_tokens, bool):
                return "ignore_special_tokens must be a boolean value"

        with lock:  # Need to get lock to keep multiple threads from hitting code
            # self.send_do_detokenize()
            if not no_log:
                print("request IP: " + str(request.remote_addr))
                # print(json.dumps(request.get_json()), flush=True)
                print("start time: ", datetime.datetime.now())

            try:
                ret_vals = detokenize_generations(
                    tokens_gpu_tensor=tokens,
                    lengths_gpu_tensor=[None] * len(tokens),
                    return_segments=False,
                    ignore_special_tokens=ignore_special_tokens
                )
                texts = ret_vals[1]
                return jsonify({"texts": texts})

            except ValueError as ve:
                return ve.args[0], 400
            print("end time: ", datetime.datetime.now())



class MegatronModifyWindowSize(Resource):
    def __init__(self, model):
        self.model = model
        print("MODEL ATTRS")
        print(model)
        print(model.__dict__) 

    @staticmethod
    def send_do_modify():
        choice = torch.tensor([MODIFY_WINDOW_SIZE_NUM], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)

    def put(self):
        args = get_args()

        if not "window_size" in request.get_json():
            return "window_size argument required", 400
        def generate_error(ws):
            return "window_size must be a list of integers of length 2, or None.  Got: " + str(ws), 400
        ws = request.get_json()["window_size"]
        if not isinstance(ws, list):
            if ws is not None:
                return generate_error(ws)
        elif len(ws) != 2:
            return generate_error(ws)
        elif not all(isinstance(x, int) for x in ws):
            return generate_error(ws) 
        no_log = not False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"

        with lock:  # Need to get lock to keep multiple threads from hitting code

            if not no_log:
                print("request IP: " + str(request.remote_addr))
                print(json.dumps(request.get_json()), flush=True)
                print("start time: ", datetime.datetime.now())

            try:
                for layer in self.model._modules["module"].decoder.layers:
                    attn = layer.self_attention.core_attention
                    attn.window_size = check_set_window_size(attn.attn_mask_type, ws)
                return jsonify({"texts": "success"})

            except ValueError as ve:
                return ve.args[0], 400
            print("end time: ", datetime.datetime.now())

class MegatronGeneralAPI(MegatronGenerate):
    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def send_do_generate():
        choice = torch.tensor([GENERAL_API_NUM], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)

    def put(self):
        raw_json = request.get_json()
        required_keys = ["prompts", "tokens_to_generate"]




class MegatronServer(object):
    def __init__(self, model):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/api', resource_class_args=[model])
        api.add_resource(MegatronTokenize, '/api/tokenize', resource_class_args=[model])
        api.add_resource(MegatronDetokenize, '/api/detokenize', resource_class_args=[model])
        api.add_resource(MegatronModifyWindowSize, '/api/modify_window_size', resource_class_args=[model])
        
    def run(self, url, port): 
        self.app.run(url, threaded=True, debug=False, port=port)