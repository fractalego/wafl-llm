import time

import numpy as np
import tritonclient.http as tritonhttpclient

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")

VERBOSE = True
input_name = 'batch_input'
output_name = 'logits'
#model_name = 'quantized_gptj_onnx'
#model_name = "gptjt_6b_onnx"
#model_name = "GPT-JT"
model_name = "gpt-j"
url = 'localhost:8000'
model_version = '1'
triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)
model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

#prompt = "                                                                                                                      Q: What is the capital of France? A:"
prompt = "Q: What is the capital of France? A:"

initial_length = len(prompt)

start = time.time()
while len(prompt) < initial_length + 10:
    input_batch = tokenizer.encode(prompt, return_tensors="np").astype(np.int32)
    input_batch = input_batch[:, -128:]
    print(input_batch.shape)
    input_ids = tritonhttpclient.InferInput(input_name, input_batch.shape, 'INT32')
    input_ids.set_data_from_numpy(input_batch, binary_data=False)
    attention_mask = tritonhttpclient.InferInput("attention_mask", input_batch.shape, 'INT32')
    mask = np.ones_like(input_batch).astype(np.int32)
    #mask[0, -1] = 0
    attention_mask.set_data_from_numpy(mask, binary_data=False)
    output = tritonhttpclient.InferRequestedOutput(output_name, binary_data=True)
    response = triton_client.infer(model_name, model_version=model_version,
                                   inputs=[input_ids, attention_mask], outputs=[output])

    logits = response.as_numpy('logits')
    new_logits = []
    #for logit in logits[0]:
    #    new_logit = []
    #    for item in logit:
    #        if item > 0:
    #            new_logit.append(-1e6)

    #        else:
    #            new_logit.append(item)

    #    new_logits.append(new_logit)

    #new_logits = np.array([new_logits])
    output_ids = np.argmax(logits, axis=-1)
    print(output_ids)
    print("**", tokenizer.decode(output_ids[0]))
    prompt += tokenizer.decode(output_ids[0][-1])

end = time.time()

print(prompt)
print("Inference time:", end - start)