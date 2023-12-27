import json
import logging
import os
import torch

from sentence_transformers import SentenceTransformer
from ts.torch_handler.base_handler import BaseHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class SentenceEmbedderHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        _logger.info("The handler is created!")
        self._config = json.load(open(os.path.join(_path, "config.json"), "r"))

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_name = self._config["sentence_embedder_models"]
        _logger.info(f"Loading the model {model_name}.")
        self._model = SentenceTransformer(model_name, device="cuda")
        _logger.info("sentence transformers model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("body").get("text")
        return {"text": text}

    def inference(self, data):
        with torch.no_grad():
            text = data["text"]
            vector = self._model.encode(text, show_progress_bar=False)
            return {"embedding": vector.tolist()}

    def postprocess(self, inference_output):
        return [inference_output]


_service = SentenceEmbedderHandler()


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
