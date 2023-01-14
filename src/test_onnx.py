import time

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
model = ORTModelForCausalLM.from_pretrained(
    "quantized_gptj_onnx", provider="TensorrtExecutionProvider"
)

# quantizer = ORTQuantizer.from_pretrained(model)
# dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
# quantizer.quantize(
#    save_dir="gptj_test",
#    quantization_config=dqconfig,
# )
# exit(0)


prompt = """
Alberto lives in San Dona di Piave. 
The task is to take out the word "London" from the following sentence and replace it to the place where Alberto lives: "Alberto went to London and had fun"
Answer:

""".strip()

tokens = tokenizer.encode(prompt, return_tensors="pt")

print("Starting to generate.")
start = time.time()
output = model.generate(
    tokens.to("cuda"),
    max_length=min(tokens.shape[1] + 15, 1023),
    pad_token_id=tokenizer.eos_token_id,
)

end = time.time()

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Time (s):", end - start)
