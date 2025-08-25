# pip install openai pydantic fastapi uvicorn
import os
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI
import json
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI


token = <GITHUB_MODEL_TOKENs>
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4o"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)


# ---------- Data models ----------
class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_suggestions: int = 4
    topic_hint: Optional[str] = None  # optionally nudge the model

class Suggestions(BaseModel):
    suggestions: List[str] = Field(
        ..., description="3–5 concise, helpful follow-up questions."
    )

    @field_validator("suggestions")
    @classmethod
    def validate_suggestions(cls, v):
        # Keep them short, unique, and non-empty
        v = [s.strip() for s in v if s and s.strip()]
        seen, deduped = set(), []
        for s in v:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(s[:140])  # keep buttons tidy
        # enforce bounds
        if not (1 <= len(deduped) <= 5):
            raise ValueError("Expect between 1 and 5 suggestions")
        return deduped

class ChatResponse(BaseModel):
    reply: str
    suggestions: List[str]

# ---------- FastAPI app ----------
app = FastAPI(title="Chat with Suggested Next Questions")

SYSTEM_SUGGESTION_INSTRUCTIONS = """\
You are a UX assistant that proposes the next best questions the user could ask.
Rules:
- Be relevant to the latest user message and overall conversation.
- Be short (≤ 12 words), specific, and non-overlapping.
- Avoid yes/no questions and avoid repeating what's already asked/answered.
- Output ONLY in JSON as per the provided schema.
"""

def generate_reply_stream(messages: List[ChatMessage]):
    """
    Generate streaming reply from AI model.
    """
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=0.7,
            stream=True
        )
        
        full_content = ""
        for chunk in stream:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    content = delta.content
                    full_content += content
                    yield f"data: {json.dumps({'type': 'content', 'data': content})}\n\n"
        
        # Generate suggestions after reply is complete
        if full_content:
            suggestions = generate_suggestions(messages + [ChatMessage(role="assistant", content=full_content)], 
                                             max_n=4, topic_hint=None)
            yield f"data: {json.dumps({'type': 'suggestions', 'data': suggestions})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        print(f"Streaming error: {e}")
        
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=0.7
        )
        content = resp.choices[0].message.content or "Sorry, I encountered an error."
        yield f"data: {json.dumps({'type': 'content', 'data': content})}\n\n"
        
        suggestions = generate_suggestions(messages + [ChatMessage(role="assistant", content=content)], 
                                         max_n=4, topic_hint=None)
        yield f"data: {json.dumps({'type': 'suggestions', 'data': suggestions})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

def generate_suggestions(messages: List[ChatMessage], max_n: int, topic_hint: Optional[str]) -> List[str]:
    """
    Ask the model to emit structured JSON for suggestions.
    """
    
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    condensed = [
        {"role": "system", "content": SYSTEM_SUGGESTION_INSTRUCTIONS},
        *({"role": "system", "content": f"Topic hint: {topic_hint}"} for _ in [0] if topic_hint),
        {"role": "user", "content": f"Latest user message:\n{last_user}\n\nReturn {min(max_n,5)} suggestions in JSON format with a 'suggestions' array."},
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=condensed,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(resp.choices[0].message.content or "{}")
        suggestions = result.get("suggestions", [])
        return suggestions[:max_n] if isinstance(suggestions, list) else []
    except:
        return ["What else can you help me with?"]

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Streaming chat endpoint"""
    return StreamingResponse(
        generate_reply_stream(req.messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", 
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Non-streaming chat endpoint (fallback)"""
    
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": m.role, "content": m.content} for m in req.messages],
        temperature=0.7
    )
    reply = resp.choices[0].message.content or "(No content)"

    suggestions = generate_suggestions(req.messages + [ChatMessage(role="assistant", content=reply)], 
                                      max_n=max(3, min(req.max_suggestions, 5)),
                                      topic_hint=req.topic_hint)

    return ChatResponse(reply=reply, suggestions=suggestions)
