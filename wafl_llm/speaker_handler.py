import base64
import json
import logging
import os
import torch
from transformers import pipeline

from ts.torch_handler.base_handler import BaseHandler
from wafl_llm.speaker_embeddings import speaker_embedding

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class SpeakerHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        _logger.info("The handler is created!")
        self._config = json.load(open(os.path.join(_path, "config.json"), "r"))

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_name = self._config["speaker_model"]
        self._device = self._config["device"]
        _logger.info(f"Loading the model {model_name}.")
        self._sinthetizer = pipeline("text-to-speech", "microsoft/speecht5_tts", device=self._device, torch_dtype=torch.bfloat16)
        _logger.info("Speaker model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("body").get("text")
        return {"text": text}

    def inference(self, data):
        with torch.no_grad():
            sample = data["text"]
            speech = self._sinthetizer(sample, forward_params={"speaker_embeddings": speaker_embedding})
            return {
                "wav": base64.b64encode(speech["audio"].tobytes()).decode("utf-8"),
                "rate": speech["sampling_rate"],
            }

    def postprocess(self, inference_output):
        return [json.dumps(inference_output)]


_service = SpeakerHandler()


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
