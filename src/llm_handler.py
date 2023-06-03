import deepspeed
import logging
import os
import re
import torch

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteria
from ts.torch_handler.base_handler import BaseHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class StopAtEOS(StoppingCriteria):
    def __init__(self, tokenizer, last_string="<|EOS|>"):
        self._tokenizer = tokenizer
        self._last_string = last_string

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        generated_text = self._tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text.rfind(self._last_string) == len(generated_text) - len(
            self._last_string
        )


class ChatbotHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        _logger.info("The handler is created!")

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_name = "mosaicml/mpt-7b-instruct"
        _logger.info(f"Loading the model {model_name}.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.truncation_side = "left"
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.init_device = "cuda"
        self.model = deepspeed.init_inference(
            AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.half,
                trust_remote_code=True,
            ),
            mp_size=1,
            dtype=torch.half,
            checkpoint=None,
            replace_method="auto",
            max_out_tokens=1024,
        ).module
        self.model.eval()
        self._stop_at_eos = StopAtEOS(self.tokenizer)
        _logger.info("Transformer model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("body").get("data")
        num_beams = data[0].get("body").get("num_beams")
        num_tokens = data[0].get("body").get("num_tokens")
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=1008
        ).cuda()
        return {
            "input_ids": input_ids,
            "num_beams": num_beams,
            "num_tokens": num_tokens,
        }

    def inference(self, data):
        with torch.no_grad():
            input_ids = data["input_ids"]
            num_beams = data["num_beams"]
            num_tokens = data["num_tokens"]
            output = self.model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                num_beams=num_beams,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=[self._stop_at_eos],
            )
            output_ids = list(output[0][input_ids.shape[1] :])
            if self.tokenizer.eos_token_id in output_ids:
                output_ids = output_ids[: output_ids.index(self.tokenizer.eos_token_id)]

            answer = self.tokenizer.decode(output_ids)
            answer = re.sub(r"(.*)<\|.*\|>(.*)", r"\1<|EOS|>", answer)
            answer = re.sub(r"(.*)#", r"\1<|EOS|>", answer)
            return answer

    def postprocess(self, inference_output):
        return [inference_output]


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
