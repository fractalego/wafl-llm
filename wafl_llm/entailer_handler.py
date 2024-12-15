import json
import logging
import os
import torch

from transformers import AutoModelForSequenceClassification
from ts.torch_handler.base_handler import BaseHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class EntailerHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        _logger.info("The Entailer handler is starting up!")
        self._config = json.load(open(os.path.join(_path, "config.json"), "r"))

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_name = self._config["entailer_model"]
        self._device = self._config["device"]
        _logger.info(f"Loading the model {model_name}.")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model.bfloat16().to(self._device)
        _logger.info(f"Entailment model {model_name} loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        print(data[0])
        lhs = data[0].get("body").get("lhs")
        rhs = data[0].get("body").get("rhs")
        return {"text1": lhs, "text2": rhs}

    def inference(self, data):
        tokenizer = self._model.tokenzier
        inputs = tokenizer(
            [self._model.config.prompt.format(**data)],
            return_tensors="pt",
            padding=True,
        ).to(self._device)
        self._model.t5.eval()
        with torch.no_grad():
            outputs = self._model.t5(**inputs)
            logits = outputs.logits
            logits = logits[:, 0, :]
            transformed_probs = torch.softmax(logits, dim=-1)
            scores = transformed_probs[:, 1]
            return {"score": float(scores[0])}

    def postprocess(self, inference_output):
        return [inference_output]


_service = EntailerHandler()


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
