import json
import logging
import os

from wafl_llm.default_handler import DefaultLLMHandler
from wafl_llm.mistral_handler import MistralHandler
from wafl_llm.phi3_4k_handler import Phi3Mini4KHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)

class LLMHandlerFactory:
    def __init__(self):
        self._config = json.load(open("config.json"))

    def get_llm_handler(self):
        handler_name = self._config["llm_model"]
        if handler_name == "fractalego/wafl-mistral_v0.1":
            _logger.info("Selected Mistral Handler")
            return MistralHandler(self._config)

        elif handler_name == "fractalego/wafl-phi3-mini-4k":
            _logger.info("Selected Phi3 Mini Handler")
            return Phi3Mini4KHandler(self._config)

        else:
            _logger.error(f"*** Unknown LLM name: {handler_name}. Using the default handler. This may cause issues. ***")
            DefaultLLMHandler(self._config)
