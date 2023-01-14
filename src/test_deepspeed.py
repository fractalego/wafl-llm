import time
import torch
import deepspeed

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel

#tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
#model = AutoModelForCausalLM.from_pretrained("quantized_gptj/")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", load_in_8bit=True, device_map="auto"
)

class ModelWrapper(GPT2LMHeadModel):
    def __init__(self, config, transformer, lm_head):
        super().__init__(config)
        self.transformer = transformer
        self.lm_head = lm_head

    def to(self, text):
        return self


model = ModelWrapper(model.config, model.transformer, model.lm_head)


prompt = """
Alberto lives in San Dona di Piave. 
The task is to take out the word "London" from the following sentence and replace it to the place where Alberto lives: "Alberto went to London and had fun"
Answer:

""".strip()
tokens = tokenizer.encode(prompt, return_tensors="pt")

ds_engine = deepspeed.init_inference(model,
                                     mp_size=1,
                                     dtype=torch.half,
                                     checkpoint=None,
                                     replace_method='auto',
                                     replace_with_kernel_inject=True)

model = ds_engine.module

start = time.time()
logits = model(
    tokens.to(device="cuda:0"),
).logits

end = time.time()
output_ids = torch.argmax(logits, dim=-1)
print(tokenizer.decode(output_ids, skip_special_tokens=True))
print("Time (s):", end - start)

prompt = """
Alberto lives in San Dona di Piave. 
The task is to take out the word "London" from the following sentence and replace it to the place where Alberto lives: "Alberto went to London and had fun"
Answer:

""".strip()
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

time.sleep(100)
