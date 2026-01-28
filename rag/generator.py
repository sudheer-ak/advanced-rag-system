import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from rag.schemas import RetrievalResult

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"  # cheap, strong for RAG

SYSTEM_PROMPT = """
You are a retrieval-augmented generation (RAG) assistant.

Rules you MUST follow:
- Only use facts from the provided context.
- If the answer is not contained in the context, say exactly:
  "I don't know based on the provided documents."
- Cite sources using [source_id] after every factual claim.
- Do NOT use outside knowledge.
- Be concise and accurate.
"""

USER_TEMPLATE = """
Context:
{context}

Question:
{question}

Answer:
"""

def build_context(results: List[RetrievalResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(f"[{r.chunk.source_id}] {r.chunk.text}")
    return "\n\n".join(blocks)

def ask_llm(question: str, results: List[RetrievalResult]) -> str:
    context = build_context(results)

    if not context.strip():
        return "I don't know based on the provided documents."

    prompt = USER_TEMPLATE.format(context=context, question=question)

    for _ in range(2):  # retry if model misbehaves
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        answer = resp.choices[0].message.content.strip()

        # basic guardrail
        if "I don't know" in answer or "[" in answer:
            return answer

    # fallback
    return "I don't know based on the provided documents."


