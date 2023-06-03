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

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_names = ["msmarco-distilbert-base-v3", "multi-qa-distilbert-dot-v1"]
        _logger.info(f"Loading the models {model_names}.")
        self._sentence_transfomers_dict = {
            "msmarco-distilbert-base-v3": SentenceTransformer(
                "msmarco-distilbert-base-v3", device="cuda"
            ),
            "multi-qa-distilbert-dot-v1": SentenceTransformer(
                "multi-qa-distilbert-dot-v1", device="cuda"
            ),
        }
        _logger.info("sentence transformers model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("body").get("text")
        model_name = data[0].get("body").get("model_name")
        return {"text": text, "model_name": model_name}

    def inference(self, data):
        with torch.no_grad():
            text = data["text"]
            model_name = data["model_name"]
            if model_name not in self._sentence_transfomers_dict:
                return {"embedding": []}

            vector = self._sentence_transfomers_dict[model_name].encode(
                text, show_progress_bar=False
            )
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
