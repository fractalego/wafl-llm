import json
import logging
import os
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimum.bettertransformer import BetterTransformer
from ts.torch_handler.base_handler import BaseHandler

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class WhisperHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        _logger.info("The handler is created!")
        self._config = json.load(open(os.path.join(_path, "config.json"), "r"))

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_name = self._config["whisper_model"]
        _logger.info(f"Loading the model {model_name}.")

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self._starting_tokens = self.processor.tokenizer.convert_tokens_to_ids(
            ["<|startoftranscript|>", "<|notimestamps|>"]
        )
        self._ending_tokens = self.processor.tokenizer.convert_tokens_to_ids(
            ["<|endoftext|>"]
        )
        self.model = (
            BetterTransformer.transform(self.model, keep_original_model=True)
            .cuda()
            .half()
        )
        self.model = torch.compile(self.model)
        _logger.info("Whisper model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        waveform = data[0].get("body").get("waveform")
        num_beams = data[0].get("body").get("num_beams")
        num_tokens = data[0].get("body").get("num_tokens")
        hotword = data[0].get("body").get("hotword")
        input_features = self.processor(
            audio=waveform, return_tensors="pt", sampling_rate=16_000
        ).input_features
        hotword_tokens = None
        if hotword:
            hotword_tokens = torch.tensor(
                [
                    item
                    for item in self.processor.tokenizer.encode(f" {hotword}")
                    if item not in set(self._ending_tokens + self._starting_tokens)
                ],
                dtype=torch.int,
            ).unsqueeze(0)

        return {
            "input_features": input_features.cuda().half(),
            "num_beams": num_beams,
            "num_tokens": num_tokens,
            "hotword_tokens": hotword_tokens.cuda().half()
            if hotword_tokens is not None
            else None,
        }

    def inference(self, data):
        with torch.no_grad():
            input_features = data["input_features"]
            num_beams = data["num_beams"]
            num_tokens = data["num_tokens"]
            hotword_tokens = data["hotword_tokens"]
            output = self.model.generate(
                input_features,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=num_tokens,
            )
            transcription = self.processor.batch_decode(
                output.sequences, skip_special_tokens=True
            )[0]
            score = output.sequences_scores
            logp = None
            if hotword_tokens is not None:
                logp = self.compute_logp(hotword_tokens, input_features)

            return {
                "transcription": transcription,
                "score": float(score),
                "logp": float(logp) if logp else None,
            }

    def postprocess(self, inference_output):
        return [json.dumps(inference_output)]

    def compute_logp(self, hotword_tokens, input_features):
        input_ids = torch.tensor([self._starting_tokens]).cuda()
        for _ in range(hotword_tokens.shape[1]):
            logits = self.model(
                input_features,
                decoder_input_ids=input_ids,
            ).logits
            new_token = torch.argmax(logits, dim=-1)
            new_token = torch.tensor([[new_token[:, -1]]]).cuda()
            input_ids = torch.cat([input_ids, new_token], dim=-1)

        logprobs = torch.log(torch.softmax(logits, dim=-1))
        sum_logp = 0
        for logp, index in zip(logprobs[0][1:], hotword_tokens):
            sum_logp += logp[int(index)]

        return sum_logp


_service = WhisperHandler()


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
