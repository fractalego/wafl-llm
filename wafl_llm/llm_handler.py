import json

import deepspeed
import logging
import os
import re
import torch

from typing import List
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteria
from ts.torch_handler.base_handler import BaseHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class StopAtEOS(StoppingCriteria):
    def __init__(self, tokenizer: "AutoTokenizer", last_strings: List[str]):
        self._tokenizer = tokenizer
        self._last_strings = last_strings

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        num_ending_tokens = 0
        for token_ids in input_ids:
            generated_text = self._tokenizer.decode(token_ids)
            for last_string in self._last_strings:
                if generated_text.endswith(last_string):
                    num_ending_tokens += 1
                    break

            if num_ending_tokens >= 1:
                return True

        return False


class ChatbotHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        _logger.info("The handler is created!")
        self._config = json.load(open(os.path.join(_path, "config.json"), "r"))

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_name = self._config["llm_model"]
        _logger.info(f"Loading the model {model_name}.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.truncation_side = "left"
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.half,
            trust_remote_code=True,
            device_map="cuda",
        )
        self.model = torch.compile(self.model)
        self.model.eval()
        _logger.info("Transformer model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("body").get("data")
        temperature = data[0].get("body").get("temperature")
        num_tokens = data[0].get("body").get("num_tokens")
        last_strings = data[0].get("body").get("last_strings")
        num_replicas = data[0].get("body").get("num_replicas")
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=8191
        ).cuda()
        return {
            "input_ids": input_ids,
            "temperature": temperature,
            "num_tokens": num_tokens,
            "last_strings": last_strings,
            "num_replicas": num_replicas,
        }

    def inference(self, data):
        with torch.no_grad():
            input_ids = data["input_ids"]
            temperature = data["temperature"]
            num_tokens = data["num_tokens"]
            last_strings = data["last_strings"]
            num_replicas = data["num_replicas"]
            print("LAST_STRINGS!", last_strings)
            stop_at_eos = StopAtEOS(self.tokenizer, last_strings)
            with torch.no_grad():
                input_ids = torch.cat([input_ids] * num_replicas, dim=0)
                output_ids = self.model.generate(
                    input_ids.cuda(),
                    max_new_tokens=num_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    stopping_criteria=[stop_at_eos],
                )
                return "<||>".join(
                    self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :])
                )

    def postprocess(self, inference_output):
        return [inference_output]


_service = ChatbotHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
