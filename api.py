from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils.cache import get_cached_response

app = FastAPI()
tokenizer = GPT2Tokenizer.from_pretrained("./npc_model")
model = GPT2LMHeadModel.from_pretrained("./npc_model")

@app.get("/generate")
def generate(prompt: str):
    cached = get_cached_response(prompt)
    if cached:
        return {"response": cached}

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
