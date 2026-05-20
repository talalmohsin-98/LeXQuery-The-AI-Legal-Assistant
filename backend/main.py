import os
from rag import get_answer
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI(
    title="LeXQuery - AI Legal Assistant",
    description="A RAG-based legal document Q&A system for Pakistan",
    version="1.0.0"
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User question cannot be empty"
    )

# Response Model
class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = get_answer(request.question)
        answer = result['answer']
        sources = result['sources']
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "LeXQuery API"
    }