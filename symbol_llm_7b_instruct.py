# Using symbol-llm-7b-instruct from this paper: https://arxiv.org/abs/2311.09278
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="Symbol-LLM/Symbol-LLM-7B-Instruct")


if 'pipe' in globals():
    del pipe
gc.collect()
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained("Symbol-LLM/Symbol-LLM-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Symbol-LLM/Symbol-LLM-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)



with open("KB.pl", "r") as f:
    prolog_kb = f.read()

question = "Based on the Prolog knowledge base above, which planets are gas giants?"


prompt = f"Context:\n{prolog_kb}\n\nQuestion:\n{question}\n\nAnswer:"


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)


response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("--- Model Response ---")
print(response[len(prompt):].strip())

# Responds with:
"""
--- Model Response ---
jupiter, saturn
"""
