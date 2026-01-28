import glob, os
from rag.ingest import pdf_to_chunks
from rag.retriever import HybridIndex

RAW = "data/raw"
INDEX = "data/index"

def main():
    pdfs = glob.glob(os.path.join(RAW, "*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in data/raw")

    all_chunks = []
    for p in pdfs:
        all_chunks.extend(pdf_to_chunks(p))

    idx = HybridIndex()
    idx.build(all_chunks)
    idx.save(INDEX)
    print(f"Indexed {len(all_chunks)} chunks from {len(pdfs)} PDFs into {INDEX}")

if __name__ == "__main__":
    main()

