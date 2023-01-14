import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.to(device="cuda:0")
model = torch.compile(
    model, passes={"triton-bmm": True}, fullgraph=True
)


prompt = """
Alberto lives in San Dona di Piave. 
The task is to take out the word "London" from the following sentence and replace it to the place where Alberto lives: "Alberto went to London and had fun"
Answer:

""".strip()

tokens = tokenizer.encode(prompt, return_tensors="pt").to(device="cuda:0")

start = time.time()
output = model.generate(
    tokens,
    max_length=min(tokens.shape[1] + 15, 1023),
    pad_token_id=tokenizer.eos_token_id,
)

end = time.time()

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Time (s):", end - start)
