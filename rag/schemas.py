from pydantic import BaseModel
from typing import List, Optional, Dict

class Chunk(BaseModel):
    source_id: str          # e.g., file.pdf:p12:c03
    doc_id: str             # e.g., file.pdf
    page: Optional[int]
    chunk_index: int
    text: str
    metadata: Dict = {}

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    method: str             # "dense", "bm25", "hybrid"

class AskRequest(BaseModel):
    question: str
    top_k: int = 6

class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    retrieved: List[RetrievalResult]

