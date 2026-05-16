from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
from threading import Thread
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

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

@app.post("/stream_complete")
def stream_complete(req: CompletionRequest):
    inputs = tokenizer([req.prompt], return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=req.max_tokens)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def generate():
        for token in streamer:
            if token.strip():
                yield f"data: {token}\n\n"
            if "\n" in token:
                break

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)