import time

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()

prompt = """
Q: What is the capital of the UK? 
A:
""".strip()

tokens = tokenizer.encode(prompt, return_tensors="pt").to(device="cuda:0")
start = time.time()
output = model.generate(
    tokens,
    max_new_tokens=10,
    pad_token_id=tokenizer.eos_token_id,
)

end = time.time()
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Time (s):", end - start)

print(model(tokens).logits)

