import json
import deepspeed
import logging
import os
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from ts.torch_handler.base_handler import BaseHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class EntailmentHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        _logger.info("The handler is created!")
        self._config = json.load(open(os.path.join(_path, "config.json"), "r"))

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_name = self._config["entailment_model"]
        _logger.info(f"Loading the model {model_name}.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        ds_engine = deepspeed.init_inference(
            AutoModelForSequenceClassification.from_pretrained(model_name),
            mp_size=1,
            dtype=torch.half,
            checkpoint=None,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # self.model = self.model.half().cuda()
        self.model = ds_engine.module
        self.model.eval()

        _logger.info("Entailment model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        premise = data[0].get("body").get("premise")
        hypothesis = data[0].get("body").get("hypothesis")
        input_ids = self.tokenizer(
            premise, hypothesis, truncation=True, return_tensors="pt"
        ).input_ids.cuda()
        return {"input_ids": input_ids}

    def inference(self, data):
        with torch.no_grad():
            input_ids = data["input_ids"]
            output = self.model(input_ids)
            return torch.softmax(output["logits"], -1)

    def postprocess(self, inference_output):
        return inference_output.tolist()


_service = EntailmentHandler()


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
