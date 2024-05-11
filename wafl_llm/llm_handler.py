import json
import logging
import os
import torch

from vllm import LLM, SamplingParams
from ts.torch_handler.base_handler import BaseHandler
from wafl_llm.variables import get_variables

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


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
        self._llm = LLM(model=model_name, dtype="bfloat16")
        _logger.info("Transformer model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        prompt = data[0].get("body").get("data")
        temperature = data[0].get("body").get("temperature")
        num_tokens = data[0].get("body").get("num_tokens")
        last_strings = data[0].get("body").get("last_strings")
        num_replicas = data[0].get("body").get("num_replicas")
        return {
            "prompt": prompt,
            "temperature": temperature,
            "num_tokens": num_tokens,
            "last_strings": last_strings,
            "num_replicas": num_replicas,
        }

    def inference(self, data):
        with torch.no_grad():
            prompt = data["prompt"]
            temperature = data["temperature"]
            num_tokens = data["num_tokens"]
            last_strings = data["last_strings"]
            num_replicas = data["num_replicas"]
            prompts = [prompt] * num_replicas
            sampling_params = SamplingParams(temperature=temperature, top_p=0.95, stop=last_strings, max_tokens=num_tokens)
            outputs = self._llm.generate(
                prompts, sampling_params
            )
            return "<||>".join(output.outputs[0].text for output in outputs)

    def postprocess(self, inference_output):
        return [json.dumps({"prediction": inference_output, "status": "success", "version": get_variables()["version"]})]


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
