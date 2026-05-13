from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

class BatchCompletionRequest(BaseModel):
    prompts: list[str]
    max_tokens: int = 128

@app.post("/complete")
def complete(req: CompletionRequest):
    result = pipe(req.prompt, 
                  max_new_tokens=req.max_tokens, 
                  do_sample=False,
                  repetition_penalty=1.3,
                  eos_token_id=pipe.tokenizer.eos_token_id)
    text = result[0]["generated_text"]
    completion = text[len(req.prompt):]
    # only take up to a newline to avoid repetition
    completion = completion.split("\n")[0].strip()
    return {"completion": completion}

@app.post("/batch_complete")
def batch_complete(req: BatchCompletionRequest):
    result = pipe(req.prompts,
                  max_new_tokens = req.max_tokens,
                  do_sample = False,
                  repetition_penalty = 1.3,
                  eos_token_id = pipe.tokenizer.eos_token_id)
    output = []
    for i, res in enumerate(result):
        text = res[0]["generated_text"]
        completion = text[len(req.prompts[i]):]
        completion = completion.split("\n")[0].strip()
        output.append(completion)
    return output

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)