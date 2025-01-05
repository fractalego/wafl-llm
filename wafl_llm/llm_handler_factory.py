import json
import logging
import os

from wafl_llm.phi3_4k_cpu_handler import Phi3Mini4KCPUHandler

from wafl_llm.default_handler import DefaultLLMHandler
from wafl_llm.llama3_handler import Llama3LLMHandler
from wafl_llm.mistral_handler import MistralHandler
from wafl_llm.phi3_4k_handler import Phi3Mini4KHandler
from wafl_llm.phi35_4k_handler import Phi35Mini4KHandler
from wafl_llm.phi35_4k_cpu_handler import Phi35Mini4KCPUHandler
from transformers import AutoConfig

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class LLMHandlerFactory:
    _handler_dictionary = {
        "cuda": {
            "wafl-mistral_v0.1": MistralHandler,
            "wafl-phi3-mini-4k": Phi3Mini4KHandler,
            "wafl-phi3.5-mini-instruct": Phi35Mini4KHandler,
            "wafl-llama-3-8B-instruct": Llama3LLMHandler,
        },
        "cpu": {
            "wafl-phi3-mini-4k": Phi3Mini4KCPUHandler,
            "wafl-phi3.5-mini-instruct": Phi35Mini4KCPUHandler,
        },
    }

    def __init__(self):
        self._config = json.load(open("config.json"))

    def get_llm_handler(self):
        model_path = self._config["llm_model"]
        device = self._config["device"]
        handler_name = AutoConfig.from_pretrained(model_path)._name_or_path
        for key in self._handler_dictionary[device].keys():
            if key in handler_name:
                _logger.info(f"Selected {key} Handler for device {device}.")
                return self._handler_dictionary[device][key](self._config)

        _logger.error(
            f"*** Unknown LLM name: {handler_name}. Using the default handler. This may cause issues. ***"
        )
        return DefaultLLMHandler(self._config)
