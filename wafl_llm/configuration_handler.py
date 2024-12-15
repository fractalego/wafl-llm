import json
import logging
import os

from ts.torch_handler.base_handler import BaseHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class ConfigurationHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        _logger.info("The Configuration handler is created!")
        self._config = json.load(open(os.path.join(_path, "config.json"), "r"))

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        self._llm_model_name = self._config["llm_model"]
        self._entailer_model_name = self._config["entailer_model"]
        self._sentence_embedder_model_name = self._config["sentence_embedder_models"]
        self._speaker_model_name = self._config["speaker_model"]
        self._whisper_model_name = self._config["whisper_model"]
        self._quantization = self._config["quantization"]
        self._device = self._config["device"]


        _logger.info("Speaker model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        version = data[0].get("body").get("version")
        return {"version": version}

    def inference(self, data):
        return {
            "llm_model": self._llm_model_name,
            "entailer_model": self._entailer_model_name,
            "sentence_embedder_models": self._sentence_embedder_model_name,
            "speaker_model": self._speaker_model_name,
            "whisper_model": self._whisper_model_name,
            "quantization": self._quantization,
            "device": self._device,
        }

    def postprocess(self, inference_output):
        return [json.dumps(inference_output)]


_service = ConfigurationHandler()


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
