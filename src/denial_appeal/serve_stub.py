"""Fallback OpenAI-compatible server.

Implements the same /v1/chat/completions surface as serve_vllm.py, backed by
the trained TF-IDF baseline (cause classification) + the template appeal
generator. Use it when no GPU / vLLM is available. All data is synthetic.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .categories import CAUSE_INFO, band_for
from .generator import _appeal_letter
from .rules import classify_rules

MODEL_NAME = "denial-appeal-finetune-baseline"
_model = None
_model_path = os.environ.get(
    "BASELINE_MODEL",
    str(Path(__file__).resolve().parents[2] / "artifacts" / "baseline_model.pkl"))

app = FastAPI(title="Denial Appeal Fine-Tune — fallback server (synthetic)")


def get_model():
    global _model
    if _model is None and Path(_model_path).exists():
        from .train_baseline import load_model
        _model = load_model(_model_path)
    return _model


def classify(text: str) -> str:
    m = get_model()
    if m is not None:
        return str(m.predict([text])[0])
    return classify_rules(text)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    temperature: float = 0.0


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": get_model() is not None, "synthetic": True}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/classify")
def classify_endpoint(payload: dict):
    text = payload.get("denial_letter", "")
    cause = classify(text)
    info = CAUSE_INFO[cause]
    return {"cause": cause, "cause_label": info["label"],
            "appeal_success_probability_band": band_for(info["base_success"]),
            "synthetic": True}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    started = time.time()
    user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    cause = classify(user)
    info = CAUSE_INFO[cause]
    code = info["denial_code"] if info["denial_code"] in user else info["denial_code"]
    appeal = _appeal_letter(cause, "SYN-CLAIM", "the service in the denial letter", code)
    content = json.dumps({
        "cause": cause,
        "appeal_success_probability_band": band_for(info["base_success"]),
        "appeal_letter": appeal,
        "note": "Synthetic proof-of-concept output. Human review required before any real use.",
    })
    return {
        "id": "chatcmpl-stub", "object": "chat.completion", "created": int(started),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(user.split()), "completion_tokens": len(content.split()),
                  "total_tokens": len(user.split()) + len(content.split())},
    }
