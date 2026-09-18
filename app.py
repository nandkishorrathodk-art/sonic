from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Tiny Sonic", version="0.1.0")


class Prompt(BaseModel):
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "tiny-sonic", "version": "0.1.0"}


@app.post("/chat")
def chat(prompt: Prompt) -> dict[str, str]:
    message = prompt.message.strip()
    lowered = message.lower()
    if lowered in {"hi", "hello", "hey"}:
        reply = "Hello. Tiny Sonic is ready."
    elif lowered in {"who are you", "what is sonic"}:
        reply = "I am Tiny Sonic, a minimal SONIC control plane."
    elif not message:
        reply = "Please provide a message."
    else:
        reply = f"Tiny Sonic received: {message}"
    return {"reply": reply}
