from fastapi import FastAPI
from rag.schemas import AskRequest, AskResponse
from rag.retriever import HybridIndex
from rag.generator import ask_llm
from rag.security import looks_like_injection

app = FastAPI(title="Advanced RAG API")

idx = HybridIndex()
idx.load("data/index")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if looks_like_injection(req.question):
        return AskResponse(
            answer="I can’t follow that request. Please ask a normal question about your documents.",
            citations=[],
            retrieved=[]
        )

    results = idx.search_hybrid(req.question, k=req.top_k)
    ans = ask_llm(req.question, results)
    cites = [r.chunk.source_id for r in results[:3]]

    return AskResponse(answer=ans, citations=cites, retrieved=results)

