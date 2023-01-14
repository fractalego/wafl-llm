from optimum.onnxruntime import ORTQuantizer
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quant_utils import QuantizationMode

from optimum.onnxruntime.configuration import QuantizationConfig

qconfig = QuantizationConfig(
    is_static=False,
    format=None,
    mode=QuantizationMode(0),  # Integer mode
    activations_dtype=QuantType.QUInt8,
    activations_symmetric=True,  # TRT only supports symmetric
    weights_dtype=QuantType.QUInt8,
    weights_symmetric=True,  # TRT only supports symmetric
    per_channel=True,
    reduce_range=False,
    nodes_to_quantize=[],
    nodes_to_exclude=[],
    qdq_add_pair_to_weight=True,
    qdq_dedicated_pair=True,
    operators_to_quantize=["torch.nn.Linear"],
)

quantizer = ORTQuantizer.from_pretrained("gpt2_onnx/")
quantizer.quantize(
    quantization_config=qconfig,
    save_dir="../gpt2_onnx_quantized",
)
