from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
from . import News_Detection
from .News_Detection import analyze_text, get_pipes, pipe1, pipe2, pipe3, are_models_loaded

app = FastAPI(
    title="Ensemble Fake News Detection API",
    description="REST API service for multi-model fake news classification",
    version="1.0.0",
)


class TextIn(BaseModel):
    text: str
    categories: Optional[str] = None


@app.get("/health")
async def health() -> Dict[str, Any]:
    models_loaded = News_Detection.are_models_loaded()
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "models": ["RoBERTa Base", "Distilled TinyBERT", "BERT-Detect"],
    }


@app.post("/api/analyze")
async def api_analyze(payload: TextIn):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Article text cannot be empty.")
    try:
        res = analyze_text(payload.text, categories=payload.categories)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
