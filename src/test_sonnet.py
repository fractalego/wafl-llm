import json
import time

import requests
from transformers import AutoTokenizer

_server_url = f"https://127.0.0.1:8080/predictions/bot"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = """
An example sonnet by Shakespeare about ageing beauty is:
<sonnet>
When in the chronicle of wasted time
I see descriptions of the fairest wights,
And beauty making beautiful old rhyme
In praise of ladies dead, and lovely knights,
Then, in the blazon of sweet beauty’s best,
Of hand, of foot, of lip, of eye, of brow,
I see their antique pen would have express’d
Even such a beauty as you master now.
So all their praises are but prophecies
Of this our time, all you prefiguring;
And, for they look’d but with divining eyes,
They had not skill enough your worth to sing:
For we, which now behold these present days,
Had eyes to wonder, but lack tongues to praise.
</sonnet>

Another sonnet by Shakespeare about love is:
<sonnet>
Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove:
O no; it is an ever-fixed mark,
That looks on tempests, and is never shaken;
It is the star to every wandering bark,
Whose worth’s unknown, although his height be taken.
Love’s not Time’s fool, though rosy lips and cheeks
Within his bending sickle’s compass come;
Love alters not with his brief hours and weeks,
But bears it out even to the edge of doom.
If this be error and upon me proved,
I never writ, nor no man ever loved.
</sonnet>

This is a sonnet about rainy days:
<sonnet>
I’ve seen it rain on sunny days
And seen the darkness flash with light
And even lightning turn to haze,
Yes, frozen snow turn warm and bright
And sweet things taste of bitterness
And what is bitter taste most sweet
And enemies their love confess
And good, close friends no longer meet.
Yet stranger things I’ve seen of love
Who healed my wounds by wounding me.
The fire in me he quenched before;
The life he gave was the end thereof,
The fire that slew eluded me.
Once saved from love, love now burns more.
</sonnet>

This is a sonnet about death:
<sonnet>
Death, be not proud, though some have called thee
Mighty and dreadful, for thou art not so;
For those whom thou think’st thou dost overthrow
Die not, poor Death, nor yet canst thou kill me.
From rest and sleep, which but thy pictures be,
Much pleasure; then from thee much more must flow,
And soonest our best men with thee do go,
Rest of their bones, and soul’s delivery.
Thou art slave to fate, chance, kings, and desperate men,
And dost with poison, war, and sickness dwell,
And poppy or charms can make us sleep as well
And better than thy stroke; why swell’st thou then?
One short sleep past, we wake eternally
And death shall be no more; Death, thou shalt die.
</sonnet>

This is a sonnet in the style of Shakespeare about a computer programmer having to work at night:
<sonnet>

""".strip()

prompt = """
'In the text below two people are discussing a story.

Story:
when the bot asks:\'Welcome to the website. How may I help you?\'

Discussion:
Q: Hello?
A:                                                '
""".strip()

tokenizer.decode([48], skip_special_tokens=True)


def predict_answer(prompt: str):
    payload = {"data": prompt, "num_beams": 1, "num_tokens": 300}
    start = time.time()
    r = requests.post(_server_url, json=payload, verify=False)
    end = time.time()
    print("Inference time: ", end - start)
    answer = json.loads(r.content.decode("utf-8"))
    start = time.time()
    print(answer)
    print("Final tokens length:", len(answer))
    print(tokenizer.decode(answer, skip_special_tokens=True))
    end = time.time()
    print("Decoding time: ", end - start)


if __name__ == "__main__":
    predict_answer(prompt)
