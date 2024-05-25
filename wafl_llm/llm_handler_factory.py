import json
import logging
import os

from wafl_llm.mistral_handler import MistralHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)

class LLMHandlerFactory:
    def __init__(self):
        self._config = json.load(open(os.path.join(_path, "config.json"), "r"))

    def get_llm_handler(self):
        handler_name = self._config["llm_model"]
        if handler_name == "fractalego/wafl-mistral_v0.1":
            _logger.info("Selected Mistral Handler")
            return MistralHandler(self._config)

        else:
            _logger.error(f"Unknown LLM name: {handler_name}")
            raise ValueError(f"Unknown LLM name: {handler_name}")
