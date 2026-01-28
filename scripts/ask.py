from rag.retriever import HybridIndex
from rag.generator import ask_llm
from rag.security import looks_like_injection

INDEX = "data/index"

def main():
    idx = HybridIndex()
    idx.load(INDEX)

    while True:
        q = input("\nQuestion (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        if looks_like_injection(q):
            print("Blocked: prompt injection detected.")
            continue

        results = idx.search_hybrid(q, k=6)
        ans = ask_llm(q, results)
        print("\nAnswer:\n", ans)

if __name__ == "__main__":
    main()

