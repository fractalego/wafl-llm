import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1").to(device="cuda:0")
#model = AutoModelForCausalLM.from_pretrained(
#    "togethercomputer/GPT-JT-6B-v1", load_in_8bit=True, device_map="auto"
#)
#model = torch.compile(
#    model, mode="reduce-overhead", fullgraph=True
#)


prompt = """
Alberto lives in San Dona di Piave. 
The task is to take out the word "London" from the following sentence and replace it to the place where Alberto lives: "Alberto went to London and had fun"
Answer:
""".strip()
#prompt = "                                                                                                                     Q: What is the capital of Italy? A:"

tokens = tokenizer.encode(prompt, return_tensors="pt")

start = time.time()
output = model.generate(
    tokens.to(device="cuda:0"),
    max_length=min(tokens.shape[1] + 15, 1023),
    pad_token_id=tokenizer.eos_token_id,
)

end = time.time()

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Time (s):", end - start)
