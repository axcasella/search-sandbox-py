from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCasualLM, pipeline, AutoModelForSeq2SeqLM

model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8_bit=True)


pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=100)
local_llm = HuggingFacePipeline(pipeline=pipe)

print(local_llm.run("Who won world cup in 1998?"))