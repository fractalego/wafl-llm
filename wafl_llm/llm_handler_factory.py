import json
import logging
import os

from wafl_llm.default_handler import DefaultLLMHandler
from wafl_llm.llama3_handler import Llama3LLMHandler
from wafl_llm.mistral_handler import MistralHandler
from wafl_llm.phi3_4k_handler import Phi3Mini4KHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class LLMHandlerFactory:
    _handler_dictionary = {
        "wafl-mistral_v0.1": MistralHandler,
        "wafl-phi3-mini-4k": Phi3Mini4KHandler,
        "wafl-phi3-mini-4k_v2": Phi3Mini4KHandler,
        "wafl-llama-3-8B-instruct": Llama3LLMHandler,
    }

    def __init__(self):
        self._config = json.load(open("config.json"))

    def get_llm_handler(self):
        handler_name = self._config["llm_model"]
        for key in self._handler_dictionary.keys():
            if key in handler_name:
                _logger.info(f"Selected {key} Handler")
                return self._handler_dictionary[key](self._config)

        _logger.error(
            f"*** Unknown LLM name: {handler_name}. Using the default handler. This may cause issues. ***"
        )
        return DefaultLLMHandler(self._config)
