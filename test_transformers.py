import transformers
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

text = "translate English to Chinese"
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
# print(translator(text))

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
token = tokenizer(text=text, return_tensors="pt")
inputs = token.input_ids

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))