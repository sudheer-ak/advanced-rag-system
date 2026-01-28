from pypdf import PdfReader
from typing import List
from rag.schemas import Chunk
from rag.chunking import chunk_text, clean_text

def extract_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(clean_text(text))
    return pages

def pdf_to_chunks(pdf_path: str) -> List[Chunk]:
    pages = extract_pdf_pages(pdf_path)
    doc_id = pdf_path.split("/")[-1]
    out: List[Chunk] = []
    for page_idx, page_text in enumerate(pages, start=1):
        pieces = chunk_text(page_text)
        for c_idx, piece in enumerate(pieces):
            source_id = f"{doc_id}:p{page_idx}:c{c_idx:02d}"
            out.append(
                Chunk(
                    source_id=source_id,
                    doc_id=doc_id,
                    page=page_idx,
                    chunk_index=c_idx,
                    text=piece,
                    metadata={}
                )
            )
    return out

