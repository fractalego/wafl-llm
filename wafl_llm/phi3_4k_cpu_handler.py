import json
import logging
import os
import torch

from typing import List
from transformers import AutoTokenizer, StoppingCriteria, AutoModelForCausalLM
from ts.torch_handler.base_handler import BaseHandler
from wafl_llm.variables import get_variables

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)
_device = "cpu"


class Phi3Mini4KCPUHandler(BaseHandler):
    def __init__(self, config):
        super().__init__()
        self.initialized = False
        _logger.info("The handler is created!")
        self._config = config
        self._last_strings = [
            "\nuser",
            "\nbot",
            "\nUser",
            "\nBot",
            "<|EOS|>",
            "</remember>",
            "</execute>\n",
            "</s>",
            "<|end|>",
            "<|assistant|>",
            "<|user|>",
            "\n\n---",
            "\n\n- output:",
            "\n\n- ai:",
            "\n\n- user:",
            "\n\n- response:",
            "[delete_rule]",
        ]

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_name = self._config["llm_model"]
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        _logger.info(f"Loading the model {model_name}.")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        _logger.info(f"Transformer model {model_name} loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("body").get("data")
        temperature = data[0].get("body").get("temperature")
        num_tokens = data[0].get("body").get("num_tokens")
        num_replicas = data[0].get("body").get("num_replicas")
        input_ids = self._get_input_ids(text).to(_device)
        return {
            "input_ids": input_ids,
            "temperature": temperature,
            "num_tokens": num_tokens,
            "last_strings": self._last_strings,
            "num_replicas": num_replicas,
        }

    def inference(self, data):
        with torch.no_grad():
            input_ids = data["input_ids"]
            temperature = data["temperature"]
            num_tokens = data["num_tokens"]
            last_strings = data["last_strings"]
            num_replicas = data["num_replicas"]
            stop_at_eos = StopAtEOS(self._tokenizer, last_strings)
            with torch.no_grad():
                input_ids = torch.cat([input_ids] * num_replicas, dim=0)
                output_ids = self._model.generate(
                    input_ids.to(_device),
                    max_new_tokens=num_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    use_cache=True,
                    stopping_criteria=[stop_at_eos],
                )
                return "<||>".join(
                    self._tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :])
                )

    def postprocess(self, inference_output):
        return [
            json.dumps(
                {
                    "prediction": inference_output,
                    "status": "success",
                    "version": get_variables()["version"],
                    "model": self._config["llm_model"],
                }
            )
        ]

    def _get_input_ids(self, chat_template_dictionary):
        chat_template_list = []
        for item in chat_template_dictionary["conversation"]:
            speaker = item["speaker"]
            text = item["text"]
            if speaker.lower() == "user":
                chat_template_list.append({"role": "user", "content": text})
            if speaker.lower() in ["assistant", "bot"]:
                chat_template_list.append({"role": "assistant", "content": text})
        input_ids = self._tokenizer.encode(
            "<|system|>\n"
            + chat_template_dictionary["system_prompt"]
            + "\n<|end|><|assistant|>\n"
        )
        input_ids = (
            input_ids + self._tokenizer.apply_chat_template(chat_template_list)[1:]
        )
        return torch.tensor([input_ids])

    def _get_system_prompt_input_ids(self, chat_template_dictionary):
        system_prompt = chat_template_dictionary["system_prompt"]
        input_ids = self._tokenizer.encode(system_prompt)
        return input_ids


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
