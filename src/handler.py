import deepspeed
import logging
import os
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from ts.torch_handler.base_handler import BaseHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class ChatbotHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        _logger.info("The handler is created!")

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        _logger.info("Loading the fine-tuned GPT-JT 6B model.")

        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1")
        ds_engine = deepspeed.init_inference(model,
                                             mp_size=1,
                                             dtype=torch.int8,
                                             checkpoint=None,
                                             replace_method='auto',
                                             replace_with_kernel_inject=True)

        self.model = ds_engine.module
        self.model.eval()

        _logger.info("Transformer model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("body").get("data")
        num_beams = data[0].get("body").get("num_beams")
        num_tokens = data[0].get("body").get("num_tokens")
        input_ids = self.tokenizer.encode(text, return_tensors="pt").cuda()
        return {"input_ids": input_ids, "num_beams": num_beams, "num_tokens": num_tokens}

    def inference(self, data):
        with torch.no_grad():
            input_ids = data["input_ids"]
            num_beams = data["num_beams"]
            num_tokens = data["num_tokens"]
            return self.model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                num_beams=num_beams,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

    def postprocess(self, inference_output):
        return inference_output.tolist()


_service = ChatbotHandler()


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
