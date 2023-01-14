import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
model = AutoModelForCausalLM.from_pretrained(
    "togethercomputer/GPT-JT-6B-v1", load_in_8bit=True, device_map="auto"
)

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

quantized_gptj_folder = "quantized_gptj"
model.save_pretrained(quantized_gptj_folder)

# quantized_model.config.save_pretrained(quantized_gptj_folder)
# quantized_state_dict = quantized_model.state_dict()
# torch.save(quantized_state_dict, os.path.join(quantized_gptj_folder, "pytorch_model.bin"))
