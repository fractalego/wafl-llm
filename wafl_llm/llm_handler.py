import logging
import os

from wafl_llm.llm_handler_factory import LLMHandlerFactory

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)

_service = LLMHandlerFactory().get_llm_handler()


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
