import time
import torch
import deepspeed

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1")

#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Q: What is the capital of Italy? A:"
tokens = tokenizer.encode(prompt, return_tensors="pt")
model.eval()
ds_engine = deepspeed.init_inference(model,
                                     mp_size=1,
                                     dtype=torch.float16,
                                     checkpoint=None,
                                     replace_method='auto',
                                     replace_with_kernel_inject=True)

model = ds_engine.module

start = time.time()
output = model(
    tokens.to(device="cuda:0"),
)

end = time.time()

print("Time (s):", end - start)
model.eval()
prompt = "Q: What is the capital of the UK? A:"
start = time.time()
tokens = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(
    tokens.to(device="cuda:0"),
    max_length=min(tokens.shape[1] + 15, 1023),
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
end = time.time()



print("Time (s):", end - start)



time.sleep(100)
