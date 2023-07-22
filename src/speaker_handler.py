import base64
import json
import logging
import os
import torch

from ts.torch_handler.base_handler import BaseHandler
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

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
        _logger.info(f"Loading the model {model_name}.")
        models, cfg, self._task = load_model_ensemble_and_task_from_hf_hub(
            model_name,
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        self._model = models[0]
        TTSHubInterface.update_cfg_with_data_cfg(cfg, self._task.data_cfg)
        self._generator = self._task.build_generator(models, cfg)
        self._generator.model.cuda()
        self._generator.vocoder.model.cuda()
        _logger.info("Speaker model loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("body").get("text")
        sample = TTSHubInterface.get_model_input(self._task, text)
        sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].cuda()
        return {"sample": sample}

    def inference(self, data):
        with torch.no_grad():
            sample = data["sample"]
            wav, rate = TTSHubInterface.get_prediction(
                self._task, self._model, self._generator, sample
            )
            return {
                "wav": base64.b64encode(wav.cpu().numpy().tobytes()).decode("utf-8"),
                "rate": rate,
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
