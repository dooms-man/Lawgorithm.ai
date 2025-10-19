# app/endpoints/check_regulation.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.endpoints.gap_detection import detect_gaps_for_regulation

router = APIRouter()

# Define a Pydantic model for each regulation chunk
class RegChunk(BaseModel):
    text: str
    page: Optional[int] = None
    file_name: Optional[str] = None

class RegChunksRequest(BaseModel):
    chunks: List[RegChunk]

@router.post("/check-regulation")
def check_regulation_endpoint(request: RegChunksRequest):
    """
    Accepts a list of regulation chunks and returns compliance gap suggestions.
    """
    reg_chunks = [chunk.dict() for chunk in request.chunks]
    suggestions = detect_gaps_for_regulation(reg_chunks)
    return {"suggestions": suggestions, "count": len(suggestions)}
