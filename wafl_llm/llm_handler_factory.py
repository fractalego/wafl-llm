import json
import os

from wafl_llm.mistral_handler import MistralHandler

_path = os.path.dirname(__file__)


class LLMHandlerFactory:
    def __init__(self):
        self._config = json.load(open(os.path.join(_path, "config.json"), "r"))

    def get_llm_handler(self):
        handler_name = self._config["llm_model"]
        if handler_name == "fractalego/wafl-mistral_v0.1":
            return MistralHandler(self._config)

        else:
            raise ValueError(f"Unknown LLM name: {handler_name}")
