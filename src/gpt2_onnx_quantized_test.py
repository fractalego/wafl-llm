import time

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2", padding=True, truncation=True)
tensorrt_model = ORTModelForCausalLM.from_pretrained(
    "gpt2_onnx/", provider="TensorrtExecutionProvider", use_cache=False
)

prompt = """
Q: What is the capital of the UK? 
A:
""".strip()

tokens = tokenizer.encode(prompt, return_tensors="pt").to(device="cuda:0")
start = time.time()
output = tensorrt_model.generate(
    tokens,
    max_new_tokens=10,
    pad_token_id=tokenizer.eos_token_id,
)

end = time.time()
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Time (s):", end - start)
