import json
import time

import requests
from transformers import AutoTokenizer


def create_text_from_summary_and_dialogue(summary, dialogue):
    text = f"""
A partial summary of the conversation is:
{summary}

With the dialogue being:
{dialogue}
    """.strip()

    return text.replace("\r\n", "\n")


_server_url = f"https://127.0.0.1:8080/predictions/bot"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

summary = """
Alberto is a customer. Alberto ordered pizza and lasagne at Dominos one hour ago. He did not order anything else.
Alberto wants to know where his pizza is. Alberto is calling Dominos to know where his orders are.
John works at Dominos. John says that his pizza is almost ready. It will be at Alberto's home in 10 minutes. 
The lasagne is already on its way with the delivery person. Unfortunately the driver had an accident and is now in the hospital.
If Alberto says "rabbit" then John replies "horse". If Alberto says "tree" John says "red".
""".strip().replace(
    "\n", " "
)

dialogue = """
John: Hello, how can I help?
""".strip()

dialogue += "\nAlberto: "

tokenizer.decode([48], skip_special_tokens=True)

def predict_answer(question: str):
    payload = {"data": question, "num_beams": 1, "num_tokens": 5}
    start = time.time()
    r = requests.post(_server_url, json=payload, verify=False)
    end = time.time()
    print("Inference time: ", end - start)
    answer = json.loads(r.content.decode("utf-8"))
    start = time.time()
    print(answer)
    print(tokenizer.decode(answer, skip_special_tokens=True))
    end = time.time()
    print("Decoding time: ", end - start)

prompt = """
Q: What is the capital of the UK? 
A:
""".strip()

if __name__ == "__main__":

    predict_answer(create_text_from_summary_and_dialogue(summary, dialogue))
    #predict_answer(prompt)
